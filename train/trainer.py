import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from att_utils import AttentionController
from tqdm import tqdm
import argparse
from torchvision.io import read_image
import torch
from typing import Dict, Any
import torch.nn as nn
import torch.optim as optim
from typing import Union
import torch.functional as F
from transformers import CLIPTokenizer, CLIPModel
import torchvision
import gc
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from datasets import load_dataset
import datasets
from att_utils import AttentionStore
from att_utils import register_attention_control, get_cross_attn_map_from_unet
from utils import save, load
import numpy as np
from data_utils.data_utils import DatasetPreprocess
from data_utils.data_utils import get_word_idx
from loss import get_grounding_loss_by_layer
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import json
from datasets import Dataset, Features, Image, ClassLabel
import os
import json
from datasets import Dataset, Features, Image, ClassLabel, Sequence, Value
import os
from torchvision.ops import roi_pool
from typing import Dict, List, Any
from torch.utils.data import Subset, DataLoader


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
resolution = 512  # 768
batch_size = 1
caption_column = 'text'
image_column = 'image'
json_path = 'train/data/metadata.jsonl'
data_dir = 'train/data/img/'  # 'train/data/train2017/'
mask_path = 'train/data/'
clip_model_name = "openai/clip-vit-base-patch16"
clip_model_name = "openai/clip-vit-large-patch14"


# train Param
clip = False
supervision = False


def get_model_size_gb(model: Union[torch.nn.Module, torch.Tensor]) -> float:
    """Calculate model size in gigabytes."""
    if isinstance(model, torch.nn.Module):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_bytes = param_size + buffer_size
    else:
        size_bytes = model.nelement() * model.element_size()

    return size_bytes / (1024 ** 3)  # Convert to GB


def print_models_size(*models, names=None):
    """Print size of multiple models."""
    if names is None:
        names = [f"Model {i+1}" for i in range(len(models))]

    for name, model in zip(names, models):
        size_gb = get_model_size_gb(model)
        print(f"{name}: {size_gb:.2f} GB")


def print_components_size(clip_encoder, tokenizer):
    """Print memory size of text encoder and tokenizer in GB"""
    import sys
    import torch

    def get_size_gb(obj):
        if isinstance(obj, torch.nn.Module):
            return sum(
                p.nelement() * p.element_size() for p in obj.parameters()
            ) / (1024**3)
        return sys.getsizeof(obj) / (1024**3)

    encoder_size = get_size_gb(clip_encoder)
    tokenizer_size = get_size_gb(tokenizer)

    print(f"Text Encoder: {encoder_size:.2f} GB")
    print(f"Tokenizer: {tokenizer_size:.2f} GB")


def load_metadata(jsonl_path: str) -> Dict[str, Dict]:
    metadata_dict = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data.get('file_name') and data.get('attn_list'):
                metadata_dict[data['file_name']] = data
    return metadata_dict


def get_mask_path(mask_base_path: str,
                  image_filename: str,
                  mask_filename: str) -> str:
    # Construire le chemin complet
    full_path = os.path.join(
        mask_base_path,
        mask_filename
    )

    return full_path


def create_enriched_dataset(filtered_dataset,
                            base_path: str,
                            metadata_path: str,
                            mask_base_path: str):
    metadata_dict = load_metadata(metadata_path)

    def generator():
        for path, label in filtered_dataset.samples:
            try:
                # Extraire le nom du fichier du chemin complet
                file_name = os.path.basename(path)

                # Récupérer les métadonnées correspondantes
                metadata = metadata_dict.get(file_name)
                if metadata is None:
                    print(f"Warning: No metadata found for {file_name}")
                    continue

                # Vérifier que attn_list existe et n'est pas vide
                if not metadata.get('attn_list'):
                    print(f"Warning: No attention list found for {file_name}")
                    continue

                # Préparer les chemins des masques de segmentation valides
                valid_masks = []
                valid_words = []

                for word, mask_filename in metadata['attn_list']:
                    if word and mask_filename:
                        # Construire le chemin complet du masque
                        full_mask_path = get_mask_path(
                            mask_base_path,
                            file_name,
                            mask_filename
                        )

                        if os.path.exists(full_mask_path):
                            valid_masks.append(full_mask_path)
                            valid_words.append(word)
                        else:
                            print(
                                f"Warning: Mask not found at {full_mask_path}"
                            )

                # Vérifier qu'il y a au moins un masque valide
                if not valid_masks:
                    print(f"Warning: No valid masks found for {file_name}")
                    continue

                yield {
                    "image_path": path,
                    "text": metadata.get('text', ''),
                    "words": valid_words,
                    "attn_list": valid_masks,
                    "label": label,
                }

            except Exception as e:
                print(f"Error processing {path}: {str(e)}")
                continue

    # Créer le dataset initial
    dataset = Dataset.from_generator(
        generator,
        features=Features({
            "image_path": Value("string"),
            "text": Value("string"),
            "words": Sequence(Value("string")),
            "attn_list": Sequence(Value("string")),
            "label": ClassLabel(num_classes=1, names=["default"])
        })
    )

    def remove_invalid_samples(dataset):
        valid_samples = []
        for example in dataset:
            if 'image_path' not in example or not example['image_path']:
                print(f"Sample with missing or invalid 'image_path' found: {example}")
                continue
            valid_samples.append(example)
        columns = {}
        for example in valid_samples:
            for key, value in example.items():
                if key not in columns:
                    columns[key] = []
                columns[key].append(value)
        return Dataset.from_dict(columns)
    # dataset = remove_invalid_samples(dataset)

    def load_images(example):
        try:
            if "image_path" not in example:
                example["image_path"] = ""
            if "attn_list" not in example:
                example["attn_list"] = []
            if "text" not in example:
                example["text"] = ""
            if "words" not in example:
                example["words"] = []
            if "label" not in example:
                example["label"] = 0
            if "valid" not in example:
                example["valid"] = True
            example["image"] = Image().encode_example(example["image_path"])
            example["masks"] = [
                Image().encode_example(mask_path)
                for mask_path in example["attn_list"]
                if os.path.exists(mask_path)
            ]

            return example
        except Exception as e:
            print(f"Error loading images for {example['image_path']}: {str(e)}")
            return None

    dataset = dataset.map(
        load_images,
        features=Features({
            "image_path": Value("string"),
            "image": Image(),
            "text": Value("string"),
            "words": Sequence(Value("string")),
            "attn_list": Sequence(Value("string")),
            "masks": Sequence(Image()),
            "label": ClassLabel(num_classes=1, names=["default"]),
            "valid": Value("bool")
        })
    )
    final_dataset = dataset.remove_columns(["valid", "image_path"])
    return final_dataset


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

# tokenizer = CLIPTokenizer.from_pretrained(
#     "CompVis/stable-diffusion-v1-4", subfolder='tokenizer'
# )
# clip_encoder: nn.Module = CLIPTextModel.from_pretrained(
#     "CompVis/stable-diffusion-v1-4", subfolder='clip_encoder'
# )

tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
clip_encoder = CLIPModel.from_pretrained(clip_model_name)

dataset_preprocess = DatasetPreprocess(
    caption_column=caption_column,
    image_column=image_column,
    train_transforms=train_transforms,
    attn_transforms=attn_transforms,
    tokenizer=tokenizer,
    train_data_dir=data_dir,
)

valid_files = set()
count = 0
with open(data_dir + "/metadata.jsonl", 'r') as f:
    for line in f:
        count += 1
        if count < 100:
            metadata = json.loads(line.strip())
            valid_files.add(metadata["file_name"])


class FilteredDataset(torchvision.datasets.DatasetFolder):
    def __init__(self, root, valid_files, transform=None):
        self.valid_files = valid_files
        self.transform = transform
        self.samples = [
            (os.path.join(root, file_name), 0)
            for file_name in valid_files
        ]

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = read_image(path)  # Charge l'image
        if self.transform:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.samples)


dataset = FilteredDataset(data_dir, valid_files)

dataset = create_enriched_dataset(
    dataset,
    metadata_path=json_path,
    base_path=data_dir,
    mask_base_path=mask_path
)

# initialize Modules and other training classes
vae: nn.Module = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder='vae'
)

noise_scheduler = DDPMScheduler()

model: nn.Module = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="unet"
)

# optimizer = optim.AdamW(
#     model.parameters(),
#     lr=lr,
#     weight_decay=weight_decay
# )
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer,
    T_max=T_max,
)
if args.checkpoint_name:
    first_epoch = load(
        model, optimizer, scheduler, 'checkpoints/' + args.checkpoint_name
    )

controller = AttentionStore()
register_attention_control(model, controller)

dataset = dataset_preprocess.preprocess(dataset)
train_data_loader = torch.utils.data.DataLoader(
    dataset,
    shuffle=True,
    batch_size=batch_size,
)

# set pretrained model to device and set nograd = True
clip_encoder.requires_grad_(False)
clip_encoder.to(device).to(weight_dtype)
vae.requires_grad_(False)
vae.to(weight_dtype)
model.requires_grad_(True)
model.to(weight_dtype)


os.system('cls' if os.name == 'nt' else 'clear')
# print_components_size(clip_encoder, tokenizer)
# print_models_size(model)
# print_models_size(vae)

# train
step_cnt = 0
global_step = 0
for epoch in range(first_epoch, last_epoch):
    model.train()
    train_loss = 0.0
    for step, batch in tqdm(enumerate(train_data_loader)):
        controller.reset()
        batch["pixel_values"] = batch["pixel_values"].to(weight_dtype).to(device)
        # print(batch["pixel_values"].size())
        vae = vae.to(device)
        latents = vae.encode(
                batch["pixel_values"]
            ).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        # print(latents.size())
        # input()
        # noise our latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        max_timestep = noise_scheduler.config.num_train_timesteps
        timesteps = torch.randint(0, max_timestep, (bsz,), device=device)
        timesteps = timesteps.long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        vae = vae.to('cpu')
        model = model.to(device)
        input_ids = batch["input_ids"].to(device)
        encoder_hidden_states = clip_encoder.text_model(input_ids)[0]
        model_pred = model(
            noisy_latents.to(weight_dtype),
            timesteps,
            encoder_hidden_states.to(weight_dtype)
        ).sample
        if supervision:
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
                # print(seg_gt.size())
                gt_seg_ls.append(seg_gt)
            attn_dict = get_cross_attn_map_from_unet(
                attention_store=controller,
                is_training_sd21=False
            )
            token_loss = 0.0
            pixel_loss = 0.0
            grounding_loss_dict = {}

            # mid_8, up_16, up_32, up_64 for sd14
            # 'mid_32',
            for layer_res in ['down_64', 'down_32', 'up_64', 'up_32']:

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

        denoise_loss = nn.functional.mse_loss(model_pred.float(),
                                              noise.float(),
                                              reduction="mean")

        # print(prompts)
        # print(word_token_idx_ls)
        # input()
        # print(gt_seg_ls)
        # input()
        # input_ids

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

        lr = scheduler.get_last_lr()[0]

        step_cnt += 1
        print('denoise', denoise_loss)
        print('grounding', grounding_loss)
        print('clip', clip_loss)

        loss = denoise_loss + grounding_loss + clip_loss

        controller.reset()

        train_loss += loss
        model = model.to('cpu')
        # Backpropagate
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        batch["pixel_values"] = batch["pixel_values"].to('cpu')
        print('cumulative', train_loss, "local", loss.item())
        if device == 'mps':
            torch.mps.empty_cache()
        gc.collect()

    global_step += 1
    train_loss = 0.0
    save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        current_epoch=epoch,
        filepath="checkpoint" + str(epoch)
    )
