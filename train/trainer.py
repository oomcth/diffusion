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
import torch.functional as F

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
train_layers_ls = list(range(40))
token_loss_scale = 1
pixel_loss_scale = 1

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
step_cnt = 0
global_step = 0
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
            for layer_res in train_layers_ls:

                attn_loss_dict = get_grounding_loss_by_layer(
                    _gt_seg_list=gt_seg_ls,
                    word_token_idx_ls=word_token_idx_ls,
                    res=layer_res,
                    input_attn_map_ls=attn_dict[layer_res],
                    is_training_sd21=False,
                )

                layer_token_loss = attn_loss_dict["token_loss"]
                layer_pixel_loss = attn_loss_dict["pixel_loss"]

                grounding_loss_dict[f"token/{layer_res}"] = layer_token_loss
                grounding_loss_dict[f"pixel/{layer_res}"] = layer_pixel_loss

                token_loss += layer_token_loss
                pixel_loss += layer_pixel_loss

            grounding_loss = token_loss_scale * token_loss
            grounding_loss += pixel_loss_scale * pixel_loss

            denoise_loss = F.mse_loss(model_pred.float(),
                                      noise.float(),
                                      reduction="mean")

            # get learing rate
            lr = scheduler.get_last_lr()[0]

            step_cnt += 1

            loss_dict = {
                "step/step_cnt": step_cnt,
                "lr/learning_rate": lr,
                "train/token_loss_w_scale": token_loss_scale * token_loss,
                "train/pixel_loss_w_scale": pixel_loss_scale * pixel_loss,
                "train/denoise_loss": denoise_loss,
                "train/total_loss": denoise_loss + grounding_loss,
            }

            # add grounding loss
            loss_dict.update(grounding_loss_dict)

            loss = denoise_loss + grounding_loss

            controller.reset()

            train_loss += loss.item()

            # Backpropagate
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1
        train_loss = 0.0

