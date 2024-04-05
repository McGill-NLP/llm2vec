# Tutorial to Use LLM2Vec Steps with Any Decoder-only Model

LLM2Vec consists of 3 simple steps to convert decoder-only LLMs into text encoders: 1) enabling bidirectional attention, 2) training with masked next token prediction, and 3) unsupervised contrastive learning. The model can be further fine-tuned with supervised data. Here, we provide a tutorial on how to use the LlaMA models. 

## 1) Enabling Bidirectional Attention

- talk about forward input attention masks

However, in order to be able to use the bidirectional attentions with all sorts of attentions, we need to create new LLaMA attention classes:
```python
class ModifiedLlamaAttention(LlamaAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False  # Initially `True` in transformers implementation


class ModifiedLlamaFlashAttention2(LlamaFlashAttention2):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False  # Initially `True` in transformers implementation


class ModifiedLlamaSdpaAttention(LlamaSdpaAttention):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False  # Initially `True` in transformers implementation

LLAMA_ATTENTION_CLASSES = {
    "eager": ModifiedLlamaAttention,  # Initially, `LlamaAttention'
    "flash_attention_2": ModifiedLlamaFlashAttention2,  # Initially, `LlamaFlashAttention2'
    "sdpa": ModifiedLlamaSdpaAttention,  # Initially, `LlamaSdpaAttention'
}
```
For now, we have changed all sorts of attention classes to non-causal (i.e., bidirectional).

```python
class ModifiedLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```
Now the `LlamaDecoderLayer` will use our implementation of `LLAMA_ATTENTION_CLASSES` instead of that in `transformers`.

Now let's build up the new decoder layer in a new LlaMA model class.
```python
class LlamaBiModel(LlamaModel):
```

We first have to use the `ModifiedLlamaDecoderLayer` in our `LlamaBiModel` class.
```python
class LlamaBiModel(LlamaModel):
    def __init__(self, config):
        LlamaPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [ModifiedLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]  # Initially, `LlamaDecoderLayer(config, layer_idx)`
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()
```

- talk about `from .attn_mask_utils import _prepare_4d_attention_mask_for_sdpa, _prepare_4d_attention_mask` and the LlamaBiModel forward function.

This is not sufficient, as transformers models use specific attention mask generation functions, `_prepare_4d_attention_mask_for_sdpa` and `_prepare_4d_attention_mask`, in the `forward` call of the `LlamaModel`. We now want to manipulate these function...


## 2) Masked Next Token Prediction

## 3.a) Unsupervised Contrastive Learning

## 3.b) Supervised Contrastive Learning
