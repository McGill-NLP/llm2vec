import torch

from packaging import version
import importlib.metadata

from transformers import GemmaModel, GemmaForCausalLM, GemmaPreTrainedModel, GemmaConfig
from transformers.models.gemma.modeling_gemma import (
    GemmaDecoderLayer,
    GemmaAttention,
    GemmaFlashAttention2,
    GemmaSdpaAttention,
    GemmaMLP,
    GemmaRMSNorm,
)

from torch import nn
from transformers.utils import logging

from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.utils.import_utils import _is_package_available
from transformers.cache_utils import Cache, StaticCache

from peft import PeftModel

logger = logging.get_logger(__name__)


def is_transformers_attn_greater_or_equal_4_41():
    if not _is_package_available("transformers"):
        return False

    return version.parse(importlib.metadata.version("transformers")) >= version.parse(
        "4.41.0"
    )


class ModifiedGemmaAttention(GemmaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedGemmaFlashAttention2(GemmaFlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedGemmaSdpaAttention(GemmaSdpaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


GEMMA_ATTENTION_CLASSES = {
    "eager": ModifiedGemmaAttention,
    "flash_attention_2": ModifiedGemmaFlashAttention2,
    "sdpa": ModifiedGemmaSdpaAttention,
}


class ModifiedGemmaDecoderLayer(GemmaDecoderLayer):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = GEMMA_ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx
        )

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class GemmaBiModel(GemmaModel):
    _no_split_modules = ["ModifiedGemmaDecoderLayer"]

    def __init__(self, config: GemmaConfig):
        if not is_transformers_attn_greater_or_equal_4_41():
            raise ValueError(
                "The current implementation of GemmaEncoderModel follows modeling_gemma.py of transformers version >= 4.41.0"
            )
        GemmaPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                ModifiedGemmaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


class GemmaBiForMNTP(GemmaForCausalLM):
    def __init__(self, config):
        GemmaPreTrainedModel.__init__(self, config)
        self.model = GemmaBiModel(config)
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
