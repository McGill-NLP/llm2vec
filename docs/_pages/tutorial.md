---
title: "Tutorial"
permalink: /tutorial/
---

# LLM2Vec Tutorial: Steps for transforming any decoder-only model into a text encoder

LLM2Vec consists of 3 simple steps to transform decoder-only LLMs into text encoders: 1) enabling bidirectional attention, 2) training with masked next token prediction, and 3) unsupervised contrastive learning. The model can be further fine-tuned with supervised data. Here, we provide a tutorial on how to use the LlaMA models.

This tutorial will focus on the first two steps. After these steps, the model can be trained for unsupervised or supervised contrastive learning like any other encoder model.

## 1) Enabling Bidirectional Attention

-  add a conceptual figure here

<!-- Will work for both Llama and Mistral -->
<!-- mention which transformer version is used for this -->

A decoder-only causal LLM consists of multiple decoder layers, each of which has a self-attention mechanism. We start bottoms-up by first modifying the attention mechanism to be bidirectional.

In order to be able to use the bidirectional attentions with all sorts of attentions, we need to create new LLaMA attention classes:
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
For now, we have changed all sorts of attention classes to non-causal (i.e., bidirectional). Next, we need to modify the decoder layer to use these new attention classes. the `__init__` function is directly copied from the `transformers` implementation of `LlamaDecoderLayer`. As `LLAMA_ATTENTION_CLASSES` point to the new attention classes, the decoder layer will use bidirectional attentions.
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
Finally, we need to modify the model class to use the new decoder layer. We create a new model class `LlamaBiModel` that inherits from `LlamaModel` and uses the new `ModifiedLlamaDecoderLayer` in its `__init__` function. Everything else remains the same as the original implementation of `LlamaModel`.
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
<!-- Llama has moved to a different  -->

This is not sufficient, as transformers models use specific attention mask generation functions, `_prepare_4d_attention_mask_for_sdpa` and `_prepare_4d_attention_mask`, in the `forward` call of the `LlamaModel`. We now want to manipulate these function...
<!-- an example to verify output? -->


## 2) Masked Next Token Prediction (MNTP)
To train our models in masked next token prediction, we again implement a wrapper model class with `LlamaBiModel` as backbone.
<!-- talk about why this is needed - point to HF script, tell the return type expected -->
```python
class BiLlamaForMNTP(LlamaForCausalLM):
```

This class will have a different `__init__` and `forward` functions as it needs special backbone model and special loss definition for MNTP.

```python
class BiLlamaForMNTP(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, attention_dropout=0.0):
        if attention_dropout > 0.0:  # Augmenting Llama model with attention dropout as there is no such parameter in the initialized LlamaConfig
            config.attention_dropout = attention_dropout
        LlamaPreTrainedModel.__init__(self, config)
        self.model = LlamaBiModel(config)  # Initially, LlamaModel
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()
```

Text about forward function and write about passing shifted tokens as labels:
```python
def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        masked_lm_loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```

<!-- talk about label shifting -->

<!-- point to other resources for simcse and supervised training, as well as pointer to our code -->
