"""
code adapted from
"""


import json
import argparse
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from accelerate import PartialState
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel

parser = argparse.ArgumentParser()

parser.add_argument("--text_file_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--model_name", type=str,
                    default="CompVis/stable-diffusion-v1-4")
parser.add_argument("--img_per_prompt", type=int, default=10)

args = parser.parse_args()

model_name = args.model_name
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

distributed_state = PartialState()
pipe.to(distributed_state.device)
pipe.set_progress_bar_config(disable=True)
print("Model loaded.")

text_file_path = args.text_file_path

with open(text_file_path, "r") as f:
    text_data = json.load(f)

# prepare the data
d = []
for index in range(len(text_data)):
    texts = [text_data[index]["text"] for _ in range(args.img_per_prompt)]
    for j in range(args.img_per_prompt):
        d.append((index, texts[j], j))
print("Data prepared.")

print("Start generating images.")
with distributed_state.split_between_processes(d) as data:
    for index, text, j in tqdm(data):
        img_id = "{}_{}".format(index, j)
        save_path = f"{args.output_dir}/{img_id}.png"
        image = pipe(prompt=[text]).images[0]
        image.save(save_path)
