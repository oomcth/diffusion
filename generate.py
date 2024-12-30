"""
Generate new samples from my model !
"""

import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
import torch
import numpy as np


clip_model_name = "openai/clip-vit-large-patch14"
vae_model_path = "CompVis/stable-diffusion-v1-4"
unet_model_path = ""

tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
text_encoder = CLIPTextModel.from_pretrained(clip_model_name)

vae = AutoencoderKL.from_pretrained(vae_model_path, subfolder="vae")

unet = UNet2DConditionModel.from_pretrained(unet_model_path)


pipe = StableDiffusionPipeline(
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    vae=vae,
    unet=unet,
    scheduler=None
)

prompt = "A cat and a wine glass"
image = pipe(prompt).images[0]

plt.imshow(np.array(image))
plt.axis("off")
plt.show()
