'''
class AttentionControl and class AttentionStore are modified from
https://github.com/google/prompt-to-prompt/blob/main/prompt-to-prompt_stable.ipynb
https://github.com/google/prompt-to-prompt/blob/main/ptp_utils.py
'''


import abc
import torch
import torch
from typing import Optional, List, Dict
import numpy as np

LOW_RESOURCE = False
SD14_TO_SD21_RATIO = 1.5


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        attn = self.forward(attn, is_cross, place_in_unet)
        # attn: 8 * res * res

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        self.step_store[key].append(attn.clone())

        return attn

    def between_steps(self):
        assert len(self.attention_store) == 0

        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


def register_attention_control(unet_model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None,temb=None,):
            is_cross = encoder_hidden_states is not None

            residual = hidden_states

            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            attention_probs = self.get_attention_scores(query, key, attention_mask)

            attention_probs = controller(attention_probs, is_cross, place_in_unet)

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = to_out(hidden_states)
            # all drop out in diffusers are 0.0
            # so we here ignore dropout

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states
        return forward

    assert controller is not None, "controller must be specified"

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    down_count = 0
    up_count = 0
    mid_count = 0

    cross_att_count = 0
    sub_nets = unet_model.named_children()
    for net in sub_nets:

        if "down" in net[0]:
            down_temp = register_recr(net[1], 0, "down")
            cross_att_count += down_temp
            down_count += down_temp
        elif "up" in net[0]:
            up_temp = register_recr(net[1], 0, "up")
            cross_att_count += up_temp
            up_count += up_temp
        elif "mid" in net[0]:
            mid_temp = register_recr(net[1], 0, "mid")
            cross_att_count += mid_temp
            mid_count += mid_temp

    controller.num_att_layers = cross_att_count


def get_cross_attn_map_from_unet(attention_store: AttentionStore, is_training_sd21, 
                                 reses=[64, 32, 16, 8], poses=["down", "mid", "up"]):
    attention_maps = attention_store.get_average_attention()

    attn_dict = {}

    if is_training_sd21:
        reses = [int(SD14_TO_SD21_RATIO * item) for item in reses]
    
    for pos in poses:
        for res in reses:
            temp_list = []
            for item in attention_maps[f"{pos}_cross"]:
                if item.shape[1] == res ** 2:
                    cross_maps = item.reshape(-1, res, res, item.shape[-1])
                    temp_list.append(cross_maps)
            # if such resolution exists
            if len(temp_list) > 0:
                attn_dict[f"{pos}_{res}"] = temp_list
    return attn_dict


class AttentionController:
    def __init__(self, unet):
        """
        Initialize the attention controller for UNet2DConditionModel
        """
        self.unet = unet
        self.attn_dict = {
            'down_64': [],
            'down_32': [],
            'mid_32': [],
            'up_32': [],
            'up_64': []
        }
        self._register_hooks()

    def _get_map_key(self, layer_name: str) -> Optional[str]:
        """Determine which attention dictionary key a layer belongs to"""
        if 'down_blocks' in layer_name:
            block_idx = int(layer_name.split('.')[1])
            return 'down_64' if block_idx < 2 else 'down_32'
        elif 'mid_block' in layer_name:
            return 'mid_32'
        elif 'up_blocks' in layer_name:
            block_idx = int(layer_name.split('.')[1])
            return 'up_32' if block_idx < 2 else 'up_64'
        return None

    def _reshape_attention_scores(self, attn_probs):
        """
        Reshape attention scores to spatial dimensions

        Args:
            attn_probs: Attention probabilities [batch, heads, sequence_length, sequence_length]
        """
        batch_size, height_width, n_tokens = attn_probs.shape
        height = width = int(np.sqrt(height_width))

        # Average across heads and reshape
        attn = attn_probs.mean(dim=1)  # [batch, height*width, n_tokens]
        attn = attn.reshape(batch_size, height, width, n_tokens)

        return attn

    def _register_hooks(self):
        """Register forward hooks on attention modules"""
        def attention_forward_hook(module, input, output):
            """Hook for capturing attention weights during the forward pass"""
            # Get the query, key, value projections
            hidden_states = input[0]
            batch_size, sequence_length, _ = hidden_states.shape

            # Get attention scores before softmax
            query = module.to_q(hidden_states)
            key = module.to_k(input[1] if len(input) > 1 else hidden_states)

            query = module.head_to_batch_dim(query)
            key = module.head_to_batch_dim(key)

            attention_scores = torch.matmul(query, key.transpose(-1, -2)) * module.scale

            # Apply softmax
            attention_probs = attention_scores.softmax(dim=-1)

            # Get the map key based on the module's name
            for name, mod in self.unet.named_modules():
                if mod == module:
                    map_key = self._get_map_key(name)
                    if map_key is not None:
                        # Reshape and store attention
                        reshaped_attn = self._reshape_attention_scores(attention_probs)
                        self.attn_dict[map_key].append(reshaped_attn.detach())
                    break

            return output

        # Register hooks for CrossAttention modules
        for name, module in self.unet.named_modules():
            if "attn" in name.lower():
                if hasattr(module, 'to_q'):  # Verify it's a CrossAttention module
                    module.register_forward_hook(attention_forward_hook)

    def reset_stores(self):
        """Clear stored attention maps"""
        for key in self.attn_dict:
            self.attn_dict[key] = []

    def get_attention_maps(self, key: str) -> List[torch.Tensor]:
        """
        Get attention maps for a specific resolution level

        Returns:
            List of attention tensors of shape [batch, height, width, n_tokens]
        """
        return self.attn_dict.get(key, [])

    def visualize_attention(self, key: str, token_idx: int = 0, batch_idx: int = 0):
        """
        Visualize attention maps for a specific token

        Args:
            key: One of 'down_64', 'down_32', 'mid_32', 'up_32', 'up_64'
            token_idx: Which token's attention to visualize
            batch_idx: Which batch item to visualize
        """
        import matplotlib.pyplot as plt

        maps = self.get_attention_maps(key)
        if not maps:
            print(f"No attention maps found for {key}")
            return

        n_maps = len(maps)
        fig, axes = plt.subplots(1, n_maps, figsize=(5*n_maps, 5))
        if n_maps == 1:
            axes = [axes]

        for idx, (ax, attn_map) in enumerate(zip(axes, maps)):
            # Get attention for specific token and batch
            attn = attn_map[batch_idx, :, :, token_idx].cpu()

            im = ax.imshow(attn, cmap='viridis')
            ax.set_title(f"{key} Layer {idx}\nToken {token_idx}")
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.show()
