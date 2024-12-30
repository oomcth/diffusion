"""
this code is adapted from :
https://github.com/mlpc-ucsd/TokenCompose/blob/main/train/src/train_token_compose.py
"""


import torch
import torch.nn as nn
from torchvision import transforms
from copy import deepcopy


# get attn loss by resolution
def get_grounding_loss_by_layer(_gt_seg_list, word_token_idx_ls, res,
                                input_attn_map_ls):

    gt_seg_list = deepcopy(_gt_seg_list)
    res = int(res[-2:])
    resize_transform = transforms.Resize((res, res))

    for i in range(len(gt_seg_list)):
        gt_seg_list[i] = resize_transform(gt_seg_list[i])
        gt_seg_list[i] = gt_seg_list[i].squeeze(0)
        gt_seg_list[i] = (gt_seg_list[i] > 0.0).float()

    token_loss = 0.0
    count = 0
    for attn_map in input_attn_map_ls:
        b, H, W, j = attn_map.shape

        for i in range(len(word_token_idx_ls)):
            obj_loss = 0.0
            single_word_idx_ls = word_token_idx_ls[i]
            mask = gt_seg_list[i].to(attn_map.device)
            if isinstance(single_word_idx_ls, list):
                count += 1
                for pos in single_word_idx_ls:
                    # ca map obj shape 8 * 16 * 16
                    ca_map_obj = attn_map[:, :, :, pos].reshape(b, H, W)
                    activation_value = \
                        (ca_map_obj * mask).reshape(b, -1).sum(dim=-1)\
                        / ca_map_obj.reshape(b, -1).sum(dim=-1)

                    obj_loss += (1.0 - torch.mean(activation_value)) ** 2

                token_loss += (obj_loss/len(single_word_idx_ls))
                del mask, activation_value
            del single_word_idx_ls

    # normalize with len words
    token_loss = token_loss / count
    # token loss end

    # pixel loss start
    # average cross attention map on different layers
    avg_attn_map_ls = []
    for i in range(len(input_attn_map_ls)):
        avg_attn_map_ls.append(
            input_attn_map_ls[i].reshape(
                -1, res, res,
                input_attn_map_ls[i].shape[-1]
            ).mean(0)
        )
    avg_attn_map = torch.stack(avg_attn_map_ls, dim=0)
    avg_attn_map = avg_attn_map.sum(0) / avg_attn_map.shape[0]
    avg_attn_map = avg_attn_map.unsqueeze(0)

    pixel_loss = 0.0
    count = 0
    for i in range(len(word_token_idx_ls)):
        word_cross_attn_ls = []
        if isinstance(word_token_idx_ls[i], list):
            count += 1
            for token_idx in word_token_idx_ls[i]:
                word_cross_attn_ls.append(
                    avg_attn_map[..., token_idx]
                )
            word_cross_attn_ls = torch.stack(
                word_cross_attn_ls, dim=0
            ).sum(dim=0)
            mask = gt_seg_list[i]
            mask = mask.to(word_cross_attn_ls.device).to(
                word_cross_attn_ls.dtype
            )
            mask = mask.mean(dim=0)
            newloss = nn.functional.cross_entropy(
                word_cross_attn_ls, mask.unsqueeze(0)
            )
            pixel_loss += newloss

    # average with len word_token_idx_ls
    pixel_loss = pixel_loss / count
    # pixel loss end

    return token_loss, pixel_loss
