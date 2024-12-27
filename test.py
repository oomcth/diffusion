import torch
from diffusers import StableDiffusionPipeline
from IPython.display import display

model_id = "mlpc-lab/TokenCompose_SD14_A"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to(device)

prompt = "A cat and a wine glass"
image = pipe(prompt).images[0]

display(image)
