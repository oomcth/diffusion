import argparse
import logging
import math
import os
import random
import shutil
import random
import itertools
import wandb

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import diffusers

from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from packaging import version
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

from attn_utils import AttentionStore
from attn_utils import register_attention_control, get_cross_attn_map_from_unet
from loss_utils import get_grounding_loss_by_layer, get_word_idx
from data_utils import DatasetPreprocess

logger = get_logger(__name__, log_level="INFO")

def main(args):
    train_layers_ls = [f"down_{res}" for res in args.train_down] + \
        [f"mid_{res}" for res in args.train_mid] + [f"up_{res}" for res in args.train_up]
    
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    mixed_precision = None

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # change output dir first
    if args.resume_from_checkpoint:
        # change the output dir manually
        resume_ckpt_number = args.resume_from_checkpoint.split("-")[-1]
        args.output_dir = f"{args.output_dir}-resume-{resume_ckpt_number}"
        logger.info(f"change output dir to {args.output_dir}")

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # register attn control to unet
    controller = AttentionStore()
    register_attention_control(unet, controller)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
        )
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    assert version.parse(accelerate.__version__) >= version.parse("0.16.0"), "accelerate 0.16.0 or above is required"

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if args.use_ema:
            ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

        for i, model in enumerate(models):
            model.save_pretrained(os.path.join(output_dir, "unet"))

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

    def load_model_hook(models, input_dir):
        if args.use_ema:
            load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
            ema_unet.load_state_dict(load_model.state_dict())
            ema_unet.to(accelerator.device)
            del load_model

        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    optimizer_cls = torch.optim.AdamW

    trained_params = unet.parameters()

    learning_rate = args.learning_rate

    optimizer = optimizer_cls(
        trained_params,
        lr=learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    attn_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )

    data_dir = args.train_data_dir

    dataset_preprocess = DatasetPreprocess(
        caption_column=args.caption_column,
        image_column=args.image_column,
        train_transforms=train_transforms,
        attn_transforms=attn_transforms,
        tokenizer=tokenizer,
        train_data_dir=data_dir,
    )

    with accelerator.main_process_first():
        logger.info(f"train data dir: {os.path.basename(args.train_data_dir)}")

        dataset = load_dataset(
            "imagefolder",
            data_dir=data_dir,
            cache_dir=args.cache_dir,
        )
        train_dataset = dataset_preprocess.preprocess(dataset["train"])

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    weight_dtype = torch.float32

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        if args.resume_from_checkpoint:
            resume_ckpt_number = args.resume_from_checkpoint.split("-")[-1]
            args.tracker_run_name = f"{args.tracker_run_name}-resume-{resume_ckpt_number}"

        init_kwargs = {
            "wandb" : {
                "name" : args.tracker_run_name
            }
        }

        accelerator.init_trackers(project_name=args.tracker_project_name, 
                                  config=tracker_config,
                                  init_kwargs=init_kwargs)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    token_loss_scale = args.token_loss_scale
    pixel_loss_scale = args.pixel_loss_scale
    is_training_sd21 = args.pretrained_model_name_or_path == "stabilityai/stable-diffusion-2-1"

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Token loss scale: {token_loss_scale}")
    logger.info(f"  Pixel loss scale: {pixel_loss_scale}")
    logger.info(f"  Is SD21: {is_training_sd21}")
    logger.info(f"  Layers w. grounding obj.: {train_layers_ls}")

    global_step = 0
    first_epoch = 0
    step_cnt = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        resume_path = args.resume_from_checkpoint

        accelerator.logger.info(f"Resuming from checkpoint {resume_path}")
        accelerator.load_state(resume_path)
        global_step = int(resume_path.split("-")[-1])

        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
        # resume step indicates how many data we should skip in this epoch

        # change step_cnt
        step_cnt = global_step * args.gradient_accumulation_steps

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):

            # we reset controller twice because we use grad_checkpointing, which will have additional forward during the backward process
            controller.reset()

            # For Resume from checkpoint, Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % 10 == 0:
                    logger.info(f"skipping data {step} / {resume_step}")
                continue

            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                max_timestep = noise_scheduler.config.num_train_timesteps
                timesteps = torch.randint(0, max_timestep, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                # add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # input ids: torch.Size([1, 77])
                input_ids = batch["input_ids"]

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(input_ids)[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                prompts = batch["text"]
                assert len(prompts) == 1, "only support batch size 1"

                postprocess_seg_ls = batch["postprocess_seg_ls"]

                word_token_idx_ls = [] # postion of token in text
                gt_seg_ls = []
                for item in postprocess_seg_ls:
                    # item: [[[words], attn_gt], [[words], attn_gt], ...]
                    # words = "teddy bear" or "surfboard" or, ....
                    words_indices = []

                    words = item[0][0]
                    words = words.lower()

                    words_indices = get_word_idx(prompts[0], words, tokenizer)

                    word_token_idx_ls.append(words_indices)

                    seg_gt = item[1] # seg_gt: torch.Size([1, 1, 512, 512])
                    gt_seg_ls.append(seg_gt)

                # calculate loss
                attn_dict = get_cross_attn_map_from_unet(
                    attention_store=controller, 
                    is_training_sd21=is_training_sd21
                )

                token_loss = 0.0
                pixel_loss = 0.0

                grounding_loss_dict = {}

                # mid_8, up_16, up_32, up_64 for sd14
                for train_layer in train_layers_ls:
                    layer_res = int(train_layer.split("_")[1])

                    attn_loss_dict = \
                        get_grounding_loss_by_layer(
                        _gt_seg_list=gt_seg_ls,
                        word_token_idx_ls=word_token_idx_ls,
                        res=layer_res,
                        input_attn_map_ls=attn_dict[train_layer],
                        is_training_sd21=is_training_sd21,
                    )

                    layer_token_loss = attn_loss_dict["token_loss"]
                    layer_pixel_loss = attn_loss_dict["pixel_loss"]

                    grounding_loss_dict[f"token/{train_layer}"] = layer_token_loss
                    grounding_loss_dict[f"pixel/{train_layer}"] = layer_pixel_loss

                    token_loss += layer_token_loss
                    pixel_loss += layer_pixel_loss

                grounding_loss = token_loss_scale * token_loss + pixel_loss_scale * pixel_loss

                denoise_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # get learing rate
                lr = lr_scheduler.get_last_lr()[0]

                step_cnt += 1

                loss_dict = {
                    "step/step_cnt" : step_cnt,
                    "lr/learning_rate" : lr,
                    "train/token_loss_w_scale": token_loss_scale * token_loss,
                    "train/pixel_loss_w_scale": pixel_loss_scale * pixel_loss,
                    "train/denoise_loss": denoise_loss,
                    "train/total_loss": denoise_loss + grounding_loss,
                }

                # add grounding loss
                loss_dict.update(grounding_loss_dict)

                if args.report_to == "wandb":
                    for name, value in loss_dict.items():
                        wandb.log({name : value}, step=step_cnt)

                loss = denoise_loss + grounding_loss

                # we reset controller twice because we use grad_checkpointing, which will have additional forward during the backward process
                controller.reset()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                                
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)

                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # save checkpoint 
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()

    accelerator.end_training()

if __name__ == "__main__":

    # put all arg parse here
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        choices=[
            "CompVis/stable-diffusion-v1-4",
            "stabilityai/stable-diffusion-2-1"
        ],
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        choices=[512, 768],
        help=(
            "res 512 for sd14, res 768 for sd21"
        ),
    )

    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )

    parser.add_argument("--num_train_epochs", type=int, default=100)

    parser.add_argument(
        "--max_train_steps",
        type=int,
        required=True,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        required=True,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )

    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=6,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        choices=[None, "wandb"],
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        required=True,
        help=(
            "Save a checkpoint of the training state every X updates."
        ),
    )

    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        required=True,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default=None
    )

    parser.add_argument(
        "--tracker_run_name",
        type=str,
        default=None
    )

    # below are additional params
    parser.add_argument(
        "--token_loss_scale", 
        type=float, 
        required=True
    )

    parser.add_argument('--train_down', nargs='+', type=int, help='use which res layers in U-Net down', default=[])
    parser.add_argument('--train_mid', nargs='+', type=int, help='use which res layers in U-Net mid', default=[])
    parser.add_argument('--train_up', nargs='+', type=int, help='use which res layers in U-Net up', default=[])

    parser.add_argument("--pixel_loss_scale", type=float, required=True)

    parser.add_argument("--caption_column", type=str, default="text")
    parser.add_argument("--image_column", type=str, default="image")

    args = parser.parse_args()

    main(args)





