import os
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import CLIPTokenizer, CLIPModel
import gc
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from att_utils import AttentionStore
from att_utils import register_attention_control, get_cross_attn_map_from_unet
from utils import save, load
from data_utils import DatasetPreprocess, get_word_idx
from data_utils import create_enriched_dataset, FilteredDataset
from loss import get_grounding_loss_by_layer
import json
from torchvision.ops import roi_pool


# parser
parser = argparse.ArgumentParser(description="Script d'entraînement de modèle")
parser.add_argument('--checkpoint_name', type=str, default=None)
args = parser.parse_args()

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
weight_dtype = torch.float16
first_epoch = 0
last_epoch = 100
train_layers_ls = list(range(40))
token_loss_scale = 1
pixel_loss_scale = 1
resolution = 512
# Batch size must be 1 (I try to implement batch_size > 1 but too much
# python loop led to poor performances.
batch_size = 1
caption_column = 'text'
image_column = 'image'
json_path = 'train/data/metadata.jsonl'
data_dir = 'train/data/img/'
mask_path = 'train/data/'
clip_model_name = "openai/clip-vit-base-patch16"
clip_model_name = "openai/clip-vit-large-patch14"


# train Param
clip = True
supervision = True
poor = False


# data preprocess
train_transforms = transforms.Compose(
    [
        transforms.Resize(resolution,
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

attn_transforms = transforms.Compose(
    [
        transforms.Resize(resolution,
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ]
)

tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
clip_encoder = CLIPModel.from_pretrained(clip_model_name)

# I have to go thought the whole coco dataset and take
#  the correctly anotated files.
# Most of the data code is in data_utils
valid_files = set()
count = 0
with open(data_dir + "/metadata.jsonl", 'r') as f:
    for line in f:
        count += 1
        if count < 100 or not poor:  # if poor just use 100 images
            metadata = json.loads(line.strip())
            valid_files.add(metadata["file_name"])


dataset = FilteredDataset(data_dir, valid_files)
dataset = create_enriched_dataset(
    dataset,
    metadata_path=json_path,
    base_path=data_dir,
    mask_base_path=mask_path
)

# other main models of the project
vae: nn.Module = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder='vae'
)

noise_scheduler = DDPMScheduler()

model: nn.Module = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="unet"
)

# Use AdamW as optimizer
optimizer = optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay
)

# Cosine scheduler  TODO warmup with fixed or decreassing lr
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer,
    T_max=T_max,
)

# We can resume training from a checkpoint
# TODO in the case where we load a checkpoint we did a useless
# init of model
if True:
    model, optimizer, scheduler, first_epoch = load(
        model, optimizer, scheduler, 'train/checkpoint3.pth'
    )

# Class that get attention from different layer of the Unet
controller = AttentionStore()
register_attention_control(model, controller)

# Main data code
dataset_preprocess = DatasetPreprocess(
    caption_column=caption_column,
    image_column=image_column,
    train_transforms=train_transforms,
    attn_transforms=attn_transforms,
    tokenizer=tokenizer,
    train_data_dir=data_dir,
)
dataset = dataset_preprocess.preprocess(dataset)
train_data_loader = torch.utils.data.DataLoader(
    dataset,
    shuffle=True,
    batch_size=batch_size,
)

# Only unet is to train. Also, set device and dtype
clip_encoder.requires_grad_(False)
clip_encoder.to(device).to(weight_dtype)
vae.requires_grad_(False)
vae.to(weight_dtype)
vae.to(device)
model.requires_grad_(True)
model.to(weight_dtype)
model.to(device)


os.system('cls' if os.name == 'nt' else 'clear')


# train
for epoch in range(first_epoch, last_epoch):
    model.train()
    train_loss = 0.0  # get average loss over epoch
    for step, batch in tqdm(enumerate(train_data_loader)):

        # forward pass
        controller.reset()
        latents = vae.encode(  # encode image into latent representation
                batch["pixel_values"].to(device).to(weight_dtype)
            ).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # generate noise
        noise = torch.randn_like(latents)
        max_timestep = noise_scheduler.config.num_train_timesteps
        timesteps = torch.randint(
            0, max_timestep, (batch_size,),
            device=device
        )
        timesteps = timesteps.long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # main model forward pass
        input_ids = batch["input_ids"].to(device)
        encoder_hidden_states = clip_encoder.text_model(input_ids)[0]
        model_pred = model(
            noisy_latents.to(weight_dtype),
            timesteps,
            encoder_hidden_states.to(weight_dtype)
        ).sample

        # token level supervision losses
        if supervision:
            prompts = batch["text"]
            postprocess_seg_ls = batch["postprocess_seg_ls"]

            word_token_idx_ls = []  # token positions
            gt_seg_ls = []  # masks
            for item in postprocess_seg_ls:
                # extract ground truth mask and token position
                words_indices = []

                words = item[0][0]
                words = words.lower()

                words_indices = get_word_idx(prompts[0], words, tokenizer)

                word_token_idx_ls.append(words_indices)

                seg_gt = item[1]
                gt_seg_ls.append(seg_gt)
            attn_dict = get_cross_attn_map_from_unet(
                attention_store=controller,
            )
            token_loss = 0.0
            pixel_loss = 0.0

            # get losses for each pertinent layers
            for layer_res in ['down_64', 'down_32', 'up_64', 'up_32']:

                attn_loss_dict = get_grounding_loss_by_layer(
                    _gt_seg_list=gt_seg_ls,
                    word_token_idx_ls=word_token_idx_ls,
                    res=layer_res,
                    input_attn_map_ls=attn_dict[layer_res],
                )

                token_loss += attn_loss_dict[0]
                pixel_loss += attn_loss_dict[1]

            grounding_loss = token_loss_scale * token_loss
            grounding_loss += pixel_loss_scale * pixel_loss

        # denoise loss
        denoise_loss = nn.functional.mse_loss(
            model_pred.float(),
            noise.float(),
            reduction="mean"
        )

        # clip loss
        if clip:
            text_embeddings = clip_encoder.text_model(input_ids=input_ids)[0]
            temperature = 0.1

            # text emb
            word_embeddings = []
            for token_indices_list in word_token_idx_ls:
                token_embeds = text_embeddings[0, token_indices_list, :]
                word_embedding = token_embeds.mean(dim=0)
                word_embeddings.append(word_embedding)
            word_embeddings = torch.stack(word_embeddings)

            # img emb
            boxes = []
            for gt_mask in gt_seg_ls:
                indices = torch.nonzero(gt_mask.squeeze(0, 1) == 1)
                min_x, min_y = indices.min(dim=0).values
                max_x, max_y = indices.max(dim=0).values
                boxes.append(torch.tensor([1, min_x.item(), min_y.item(),
                                          max_x.item(), max_y.item()]))
            boxes = torch.stack(boxes).squeeze(1).to(device,
                                                     dtype=weight_dtype)
            image_embeddings = roi_pool(input=latents-model_pred,
                                        boxes=boxes,
                                        output_size=(28, 28))
            image_embeddings = image_embeddings.mean(dim=1)
            image_embeddings = image_embeddings.view(
                image_embeddings.size(0), 784
            )
            image_embeddings = image_embeddings[:, :768]

            norms = image_embeddings.norm(p=2, dim=1, keepdim=True)
            norms = torch.where(norms == 0, torch.ones_like(norms), norms)
            image_embeddings_normalized = image_embeddings / norms
            word_embeddings = torch.nn.functional.normalize(word_embeddings,
                                                            dim=1)

            logits = torch.matmul(
                image_embeddings, word_embeddings.T
            ) / temperature
            labels = torch.arange(logits.size(0), device=logits.device)
            loss_t2r = torch.nn.functional.cross_entropy(logits, labels)
            loss_r2t = torch.nn.functional.cross_entropy(logits.T, labels)
            clip_loss = (loss_t2r + loss_r2t) / 2.0

        print('denoise', denoise_loss)
        if supervision:
            print('grounding', grounding_loss)
        else:
            grounding_loss = 0
        if clip:
            print('clip', clip_loss)
        else:
            clip_loss = 0

        loss = denoise_loss + grounding_loss + clip_loss
        loss.backward()
        controller.reset()

        train_loss += loss / len(train_data_loader)

        # Backpropagate
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        print('cumulative', train_loss, "local", loss.item())
        gc.collect()

    train_loss = 0.0
    save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        current_epoch=epoch,
        filepath="checkpoint" + str(epoch) + ".pth"
    )
