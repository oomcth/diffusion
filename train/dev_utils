import torch
from typing import Union


def get_model_size_gb(model: Union[torch.nn.Module, torch.Tensor]) -> float:
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

    return size_bytes / (1024 ** 3)


def print_models_size(*models, names=None):
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
