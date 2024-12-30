import torch
from torch import nn
from torchvision.ops import roi_pool
from torch.nn.functional import adaptive_avg_pool2d

# Example of latents-model_pred tensor of size [1, 1, 512, 512]
latents_model_pred = torch.randn(1, 1, 512, 512)  # Random example, replace with your actual tensor

# Example of bounding boxes, each row represents [xmin, ymin, xmax, ymax]
# Assuming pixel coordinates, and your image size is 512x512
boxes = torch.tensor([[100, 100, 400, 400]])  # A single bounding box for demonstration

# Normalize the boxes to [0, 1] range
image_height, image_width = 512, 512
boxes = boxes.float()  # Ensure it's a float tensor for normalization
boxes[:, [0, 2]] /= image_width  # Normalize xmin and xmax
boxes[:, [1, 3]] /= image_height  # Normalize ymin and ymax

# Use roi_pool (assuming you're using PyTorch)
output_size = (24, 24)
pooled_regions = roi_pool(latents_model_pred, boxes, output_size)

print(pooled_regions.shape)  # Expected shape: [batch_size, channels, 24, 24]
