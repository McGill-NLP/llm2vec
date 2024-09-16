import torch

from packaging import version
import importlib.metadata

from transformers import LlamaModel, LlamaForCausalLM, LlamaPreTrainedModel, LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaSdpaAttention,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

from torch import nn
from transformers.utils import logging
from transformers.cache_utils import Cache, StaticCache

from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.utils.import_utils import _is_package_available

from peft import PeftModel

logger = logging.get_logger(__name__)


def is_transformers_attn_greater_or_equal_4_43_1():
    if not _is_package_available("transformers"):
        return False

    return version.parse(importlib.metadata.version("transformers")) >= version.parse(
        "4.43.1"
    )


def is_transformers_attn_greater_or_equal_4_40():
    if not _is_package_available("transformers"):
        return False

    return version.parse(importlib.metadata.version("transformers")) >= version.parse(
        "4.40.0"
    )


class ModifiedLlamaAttention(LlamaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedLlamaFlashAttention2(LlamaFlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedLlamaSdpaAttention(LlamaSdpaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


LLAMA_ATTENTION_CLASSES = {
    "eager": ModifiedLlamaAttention,
    "flash_attention_2": ModifiedLlamaFlashAttention2,
    "sdpa": ModifiedLlamaSdpaAttention,
}


class ModifiedLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx
        )

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class LlamaBiModel(LlamaModel):
    _no_split_modules = ["ModifiedLlamaDecoderLayer"]

    def __init__(self, config: LlamaConfig):
        if not is_transformers_attn_greater_or_equal_4_43_1():
            raise ValueError(
                "The current implementation of LlamaEncoderModel follows modeling_llama.py of transformers version >= 4.38.0"
            )
        LlamaPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                ModifiedLlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def _update_causal_mask(
        self,
        attention_mask,
        input_tensor,
        cache_position,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        # if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
        #     if AttentionMaskConverter._ignore_causal_mask_sdpa(
        #         attention_mask,
        #         inputs_embeds=input_tensor,
        #         past_key_values_length=past_seen_tokens,
        #         is_training=self.training,
        #     ):
        #         return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = torch.zeros(
            (sequence_length, target_length), dtype=dtype, device=device
        )  # in original implementation - torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        # Commenting out next 2 lines to disable causal masking
        # if sequence_length != 1:
        #     causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(
            target_length, device=device
        ) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(
            input_tensor.shape[0], 1, -1, -1
        )
        if attention_mask is not None:
            causal_mask = (
                causal_mask.clone()
            )  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[
                    :, None, None, :
                ].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[
                    ..., :mask_length
                ].masked_fill(padding_mask, min_dtype)
            elif attention_mask.dim() == 4:
                # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
                # cache. In that case, the 4D attention mask attends to the newest tokens only.
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0],
                    : mask_shape[1],
                    offset : mask_shape[2] + offset,
                    : mask_shape[3],
                ] = mask_slice

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        return causal_mask


class LlamaBiForMNTP(LlamaForCausalLM):
    def __init__(self, config):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = LlamaBiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    # getter for PEFT model
    def get_model_for_peft(self):
        return self.model

    # setter for PEFT model
    def set_model_for_peft(self, model: PeftModel):
        self.model = model

    # save the PEFT model
    def save_peft_model(self, path):
        self.model.save_pretrained(path)
