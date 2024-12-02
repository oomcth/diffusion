import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import typing as tp
from accelerate import Accelerator
import torch
import torch.nn as nn
import torch.optim as optim
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from att_utils import AttentionStore
from att_utils import register_attention_control, get_cross_attn_map_from_unet
from data_utils.data_utils import DatasetPreprocess
from datasets import load_dataset
from transformers import CLIPTextModel, CLIPTokenizer
from data_utils.data_utils import get_grounding_loss_by_layer, get_word_idx


# set better device
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'


# set parameter TODO use parser instead
T_max = 50
lr = 1e-4
weight_decay = 1e-4
weight_dtype = torch.float32
first_epoch = 0
last_epoch = 100


# initialize Modules and other training classes
model: nn.Module = UNet2DConditionModel(
    "CompVis/stable-diffusion-v1-4",
)

text_encoder: nn.Module = CLIPTextModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4"
)

tokenizer = CLIPTokenizer.from_pretrained(
    "CompVis/stable-diffusion-v1-4"
)

vae: nn.Module = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4"
)

noise_scheduler = DDPMScheduler.from_pretrained(
    "CompVis/stable-diffusion-v1-4"
)

optimizer = optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer,
    T_max=T_max,
)

controller = AttentionStore()


# data preprocess
data_preprocess = DatasetPreprocess()
dataset = load_dataset(
            "train/data",
            data_dir="data_dir/",
            cache_dir="cache_dir/",
        )
train_data_loader = data_preprocess(dataset)


# set pretrained model to device and set nograd = True
text_encoder.requires_grad_(False)
text_encoder.to(device, dtype=weight_dtype)
vae.requires_grad_(False)
vae.to(device, dtype=weight_dtype)


# train
for epoch in range(first_epoch, last_epoch):
    model.train()
    train_loss = 0.0
    for step, batch in enumerate(train_data_loader):
        latents = vae.encode(
            vae.encode(
                batch["pixel_values"].to(weight_dtype)
            ).latent_dist.sample()
        )
        latents = latents * vae.config.scaling_factor

        # noise our latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        max_timestep = noise_scheduler.config.num_train_timesteps
        timesteps = torch.randint(0, max_timestep, (bsz,), device=device)
        timesteps = timesteps.long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        input_ids = batch["input_ids"]
        encoder_hidden_states = text_encoder(input_ids)[0]
        model_pred = model(
            noisy_latents, timesteps, encoder_hidden_states
        ).sample

        prompts = batch["text"]
        postprocess_seg_ls = batch["postprocess_seg_ls"]

        word_token_idx_ls = []
        gt_seg_ls = []
        for item in postprocess_seg_ls:
            words_indices = []

            words = item[0][0]
            words = words.lower()

            words_indices = get_word_idx(prompts[0], words, tokenizer)

            word_token_idx_ls.append(words_indices)

            seg_gt = item[1]
            gt_seg_ls.append(seg_gt)
            attn_dict = get_cross_attn_map_from_unet(
                attention_store=controller,
                is_training_sd21=False
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
