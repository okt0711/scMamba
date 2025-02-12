# Copyright (c) 2023, Albert Gu, Tri Dao.

import math
from functools import partial
import json
import os

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mamba_ssm.models.config_mamba import MambaConfig
from src.models.mamba_ssm.modules.mamba_simple import Mamba, Block
from src.models.mamba_ssm.utils.generation import GenerationMixin
from src.models.mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from src.models.mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        vocab_size: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
        is_cls=False,
        is_mask=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.is_cls = is_cls

        self.expression_embedding = nn.Linear(1, d_model, **factory_kwargs)
        self.gene_id_embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        if is_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model, **factory_kwargs))
        if is_mask:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model, **factory_kwargs))

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, expression, gene_ids, masked_indices=None, inference_params=None):
        expression_embedding = self.expression_embedding(expression.unsqueeze(2))

        ### w/ cls token
        if masked_indices is not None:
            expression_embedding[masked_indices] = self.mask_token.to(expression_embedding.dtype)
        if self.is_cls:
            cls_token = self.cls_token.expand(gene_ids.size(0), -1, -1)
            len_embedding = expression_embedding.size(1)
            expression_embedding = torch.cat((cls_token, expression_embedding[:, :len_embedding // 2], cls_token, expression_embedding[:, len_embedding // 2:], cls_token), dim=1)

        gene_id_embedding = self.gene_id_embedding(gene_ids)

        hidden_states = gene_id_embedding + expression_embedding

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states


class MambaModel(nn.Module, GenerationMixin):

    def __init__(
        self,
        d_model: int = 2560,
        n_layer: int = 64,
        vocab_size: int = 50277,
        ssm_cfg: dict = {},
        rms_norm: bool = True,
        residual_in_fp32: bool = True,
        fused_add_norm: bool = True,
        pad_vocab_size_multiple: int = 8,
        initializer_cfg=None,
        device=None,
        dtype=None,
        is_cls=False,
        is_mask=True
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            is_cls=is_cls,
            is_mask=is_mask,
            **factory_kwargs,
        )

        self.regressor = nn.Linear(d_model, 1, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, expression, gene_ids, masked_indices=None, return_embeds=False, position_ids=None, inference_params=None, num_last_tokens=0, state=None):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(expression, gene_ids, masked_indices, inference_params=inference_params)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        if return_embeds:
            return hidden_states, None
        output = self.regressor(hidden_states)[:, :, 0]
        return output, None

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f)


class MambaCLSHeadModel(nn.Module, GenerationMixin):

    def __init__(
        self,
        d_model: int = 2560,
        n_layer: int = 64,
        vocab_size: int = 50277,
        ssm_cfg: dict = {},
        rms_norm: bool = True,
        residual_in_fp32: bool = True,
        fused_add_norm: bool = True,
        pad_vocab_size_multiple: int = 8,
        initializer_cfg=None,
        device=None,
        dtype=None,
        is_cls=True,
        is_mask=False,
        num_class=2
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        self.is_cls = is_cls

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            is_cls = is_cls,
            is_mask=is_mask,
            **factory_kwargs,
        )

        self.cls_head = nn.Linear(d_model * 3, num_class, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, expression, gene_ids, masked_indices=None, position_ids=None, inference_params=None, state=None):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(expression, gene_ids, masked_indices, inference_params=inference_params)
        if self.is_cls:
            len_hidden = hidden_states.size(1) - 1
            hidden_states = hidden_states[:, [0, len_hidden // 2, -1], :]
            hidden_states = hidden_states.view(hidden_states.size(0), -1)
        else:
            hidden_states = torch.mean(hidden_states, dim=1, keepdim=False)
        logits = self.cls_head(hidden_states)
        return logits, None

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f)


def load_backbone(model, state_dict, freeze_backbone=False, ignore_head=True, ignore_token=True):
    """

    Modifies state dict loading with custom function.  Every layer in new model will be

    inputs:
        model: nn.Module, the from 'scratch' model
        state_dict: dict, from the pretrained weights
        ignore_head: bool, whether to inflate weights in the head (or keep scratch weights).
            If number of classes changes (eg, imagenet to hmdb51), then you need to use this.

    return:
        state_dict: dict, update with inflated weights
    """

    # consumes prefix from pretrained model, if necessary
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        state_dict, "model."
    )

    model_new_params_dict = model.state_dict()
    updated_model_state_dict = {}

    # loop through scratch model keys (pretrained may have extra stuff)
    for key in sorted(model_new_params_dict.keys()):

        loaded_params = state_dict.get(key, None)
        # make sure key is in the loaded params first, if not, then print it out
    
        # if loaded_params is None:
            # This should never happen, it should be there!
            # print("Missing key in pretrained model!", key)
            # raise Exception

        if ignore_head and 'head' in key:
            # ignore head weights
            print("found head key / parameter, load from scratch", key)
            # using scratch by default, nothing needed
            used_params = model_new_params_dict[key]

        elif ignore_token and 'token' in key:
            # ignore head weights
            print("found token key / parameter, load from scratch", key)
            # using scratch by default, nothing needed
            used_params = model_new_params_dict[key]

        elif "decoder" in key:
            print("found decoder key / parameter, load from scratch", key)
            used_params = model_new_params_dict[key]
        else:
            print('key: shape MATCH, loading', key)  # load matched weights
            used_params = loaded_params

        # we need to pass back a state dict with the '.model' prefix!!!!!
        key_with_prefix = 'model.' + key
        updated_model_state_dict[key_with_prefix] = used_params

    if freeze_backbone:
        print("freezing model backbone params!")
        # note, decoder not included in backbone
        for name, param in model.named_parameters():
            param.requires_grad = False

    # we have updated the new model state dict with pretrained now
    return updated_model_state_dict
