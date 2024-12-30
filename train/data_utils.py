"""

Overall the way I load my data is suboptimal, I missunderstood some
hugging face documentation. I could have do what this file does in
a way more effiscient way.

Still my methods gives descent performances and is acceptable for the project.

"""


import os
import numpy as np
from typing import Dict
import json
from datasets import Dataset, Features, ClassLabel, Sequence, Value
import datasets
import torchvision
from torchvision.io import read_image


def get_word_idx(text: str, tgt_word, tokenizer):

    tgt_word = tgt_word.lower()

    encoded_text = tokenizer.encode(text)[1:-1]
    encoded_tgt_word = tokenizer.encode(tgt_word)[1:-1]

    first_token_idx = -1
    for i in range(len(encoded_text)):
        if encoded_text[i] == encoded_tgt_word[0]:

            if len(encoded_text) > 0:
                following_match = True
                for j in range(1, len(encoded_tgt_word)):
                    if encoded_text[i + j] != encoded_tgt_word[j]:
                        following_match = False
                if not following_match:
                    continue
            first_token_idx = i

            break

    if first_token_idx == -1:
        # would be usefull if we try to have
        # greater than one batch size
        return -1

    tgt_word_tokens_idx_ls = [i + 1 + first_token_idx
                              for i in range(len(encoded_tgt_word))]

    encoded_text = tokenizer.encode(text)

    decoded_token_ls = []

    for word_idx in tgt_word_tokens_idx_ls:
        text_decode = tokenizer.decode([encoded_text[word_idx]]).strip("#")
        decoded_token_ls.append(text_decode)

    decoded_tgt_word = "".join(decoded_token_ls)

    tgt_word_ls = tgt_word.split(" ")
    striped_tgt_word = "".join(tgt_word_ls).strip("#")

    assert decoded_tgt_word == striped_tgt_word

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
                    f"Caption column `{self.caption_column}` must"
                    f"contain strings or lists of strings."
                )

        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    def data_preprocess_train(self, examples):

        images = [image.convert("RGB")
                  for image in examples[self.image_column]]

        examples["pixel_values"] = np.array(
            [self.train_transforms(image) for image in images]
        )

        examples["input_ids"] = self.tokenize_captions(examples)

        examples["postprocess_seg_ls"] = []
        maxi = 0

        # Would be usefull if we do batch_size > 1
        # for now its completely useless
        # def pad_words(words, max_words=4, padding_word="rien"):
        #     num_words = len(words)
        #     for _ in range(max_words-num_words):
        #         words.append(padding_word)
        #     words = words[:max_words]
        #     return words

        # def pad_masks(masks,
        #               max_masks=4,
        #               mask_shape=(1, 512, 512)):
        #     num_masks = len(masks)
        #     for _ in range(max_masks-num_masks):
        #         masks.append(torch.zeros(mask_shape))
        #     masks = masks[:max_masks]
        #     return masks

        for i in range(len(examples["attn_list"])):
            maxi = max(maxi, len(examples["attn_list"][i]))

            assert len(examples["attn_list"][i]) == len(examples["words"][i])
            attn_list = examples["attn_list"][i]
            postprocess_attn_list = []
            for j, _ in enumerate(attn_list):
                category = examples["words"][i][j]

                attn_gt = examples["masks"][i][j]

                attn_gt = self.attn_transforms(attn_gt)

                if attn_gt.shape[0] == 4:
                    attn_gt = attn_gt[0].unsqueeze(0)

                if attn_gt.max() > 0:
                    attn_gt = attn_gt / attn_gt.max()

                postprocess_attn_list.append([
                    category,
                    attn_gt
                ])
            examples["postprocess_seg_ls"].append(postprocess_attn_list)
        del examples["image"]
        del examples["attn_list"]
        del examples["masks"]
        del examples["label"]
        return examples

    def preprocess(self, input_dataset):
        return input_dataset.with_transform(self.data_preprocess_train)


# load json
def load_metadata(jsonl_path: str) -> Dict[str, Dict]:
    metadata_dict = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data.get('file_name') and data.get('attn_list'):
                metadata_dict[data['file_name']] = data
    return metadata_dict


# get mask files loc
def get_mask_path(mask_base_path: str,
                  image_filename: str,
                  mask_filename: str) -> str:
    full_path = os.path.join(
        mask_base_path,
        mask_filename
    )

    return full_path


# add metadata to dataset
def create_enriched_dataset(filtered_dataset,
                            base_path: str,
                            metadata_path: str,
                            mask_base_path: str):
    metadata_dict = load_metadata(metadata_path)

    def generator():  # I debugged on my PC that has low RAM so I use generator
        for path, label in filtered_dataset.samples:
            try:
                file_name = os.path.basename(path)

                metadata = metadata_dict.get(file_name)
                if metadata is None:
                    print(f"Warning: No metadata found for {file_name}")
                    continue

                if not metadata.get('attn_list'):
                    print(f"Warning: No attention list found for {file_name}")
                    continue

                valid_masks = []
                valid_words = []

                for word, mask_filename in metadata['attn_list']:
                    if word and mask_filename:
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
            example["image"] = datasets.Image(
                ).encode_example(example["image_path"])
            example["masks"] = [
                datasets.Image().encode_example(mask_path)
                for mask_path in example["attn_list"]
                if os.path.exists(mask_path)
            ]

            return example
        except Exception as e:
            print(f"Error loading images {example['image_path']}: {str(e)}")
            return None

    dataset = dataset.map(
        load_images,
        features=Features({
            "image_path": Value("string"),
            "image": datasets.Image(),
            "text": Value("string"),
            "words": Sequence(Value("string")),
            "attn_list": Sequence(Value("string")),
            "masks": Sequence(datasets.Image()),
            "label": ClassLabel(num_classes=1, names=["default"]),
            "valid": Value("bool")
        })
    )
    print(dataset)
    final_dataset = dataset.remove_columns(["valid", "image_path"])
    return final_dataset


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
        image = read_image(path)
        if self.transform:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.samples)
