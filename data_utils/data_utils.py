'''
The following code is partly adapted from
https://github.com/mlpc-ucsd/TokenCompose/blob/5633f816116fdf7de74d9055ff9666c5222416e2/train/src/data_utils.py
'''
from PIL import Image
import os
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from copy import deepcopy

SD14_TO_SD21_RATIO = 1.5


# get token index in text
def get_word_idx(text: str, tgt_word, tokenizer):

    tgt_word = tgt_word.lower()

    # ignore the first and last token
    encoded_text = tokenizer.encode(text)[1:-1]
    encoded_tgt_word = tokenizer.encode(tgt_word)[1:-1]

    # find the idx of target word in text
    first_token_idx = -1
    for i in range(len(encoded_text)):
        if encoded_text[i] == encoded_tgt_word[0]:

            if len(encoded_text) > 0:
                # check the following 
                following_match = True
                for j in range(1, len(encoded_tgt_word)):
                    if encoded_text[i + j] != encoded_tgt_word[j]:
                        following_match = False
                if not following_match:
                    continue
            # for a single encoded idx, just take it
            first_token_idx = i

            break

    if first_token_idx == -1:
        return -1

    # add 1 for sot token
    tgt_word_tokens_idx_ls = [i + 1 + first_token_idx for i in range(len(encoded_tgt_word))]

    # sanity check
    encoded_text = tokenizer.encode(text)

    decoded_token_ls = []

    for word_idx in tgt_word_tokens_idx_ls:
        text_decode = tokenizer.decode([encoded_text[word_idx]]).strip("#")
        decoded_token_ls.append(text_decode)

    decoded_tgt_word = "".join(decoded_token_ls)

    tgt_word_ls = tgt_word.split(" ")
    striped_tgt_word = "".join(tgt_word_ls).strip("#")

    assert decoded_tgt_word == striped_tgt_word, "decode_text != striped_tar_wd"

    return tgt_word_tokens_idx_ls


class DatasetPreprocess:
    def __init__(self, caption_column, image_column, train_transforms,
                 attn_transforms, tokenizer, train_data_dir,
                 segment_dir_origin_path="seg",
                 segment_dir_relative_path="../coco_gsam_seg"):
        self.caption_column = caption_column
        self.image_column = image_column

        self.train_transforms = train_transforms
        self.attn_transforms = attn_transforms

        self.tokenizer = tokenizer

        self.train_data_dir = train_data_dir
        self.segment_dir_origin_path = segment_dir_origin_path
        self.segment_dir_relative_path = segment_dir_relative_path

    def tokenize_captions(self, examples):
        captions = []

        for caption in examples[self.caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            else:
                raise ValueError(
                    f"Caption column `{self.caption_column}` must contain strings or lists of strings."
                )

        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    def data_preprocess_train(self, examples):

        images = [image.convert("RGB") for image in examples[self.image_column]]

        examples["pixel_values"] = np.array([self.train_transforms(image) for image in images])
        # process text
        examples["input_ids"] = self.tokenize_captions(examples)

        # read in attn gt, for hard attn map
        examples["postprocess_seg_ls"] = []
        maxi = 0

        def pad_words(words, max_words=4, padding_word="rien"):
            # Pad the list of words to ensure it has exactly `max_words` words
            num_words = len(words)
            for _ in range(max_words-num_words):
                words.append(padding_word)
            words = words[:max_words]
            return words

        def pad_masks(masks,
                      max_masks=4,
                      mask_shape=(1, 512, 512)):
            num_masks = len(masks)
            for _ in range(max_masks-num_masks):
                masks.append(torch.zeros(mask_shape))
            masks = masks[:max_masks]
            return masks

        for i in range(len(examples["attn_list"])):
            maxi = max(maxi, len(examples["attn_list"][i]))

            assert len(examples["attn_list"][i]) == len(examples["words"][i])
            attn_list = examples["attn_list"][i]
            postprocess_attn_list = []
            for j, _ in enumerate(attn_list):
                # if j > 4:
                #     break
                category = examples["words"][i][j]

                attn_gt = examples["masks"][i][j]

                attn_gt = self.attn_transforms(attn_gt)

                if attn_gt.shape[0] == 4:
                    # 4 * 512 * 512 -> 1 * 512 * 512
                    attn_gt = attn_gt[0].unsqueeze(0)

                # normalize to [0, 1]
                if attn_gt.max() > 0:
                    attn_gt = attn_gt / attn_gt.max()

                postprocess_attn_list.append([
                    category,
                    attn_gt
                ])
            # postprocess_attn_list = list(
            #     zip(
            #         pad_words([word for word, _ in postprocess_attn_list]),
            #         pad_masks([mask for _, mask in postprocess_attn_list])
            #     )
            # )
            examples["postprocess_seg_ls"].append(postprocess_attn_list)
        del examples["image"]
        del examples["attn_list"]
        del examples["label"]
        del examples["masks"]
        del examples["words"]

        return examples

    def preprocess(self, input_dataset):
        return input_dataset.with_transform(self.data_preprocess_train)
