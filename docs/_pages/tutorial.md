---
title: "LLM2Vec Tutorial: Steps for transforming any decoder-only model into a text encoder"
permalink: /tutorial/
---

LLM2Vec consists of 3 simple steps to transform decoder-only LLMs into text encoders: 1) enabling bidirectional attention, 2) training with masked next token prediction, and 3) unsupervised contrastive learning. The model can be further fine-tuned with supervised data. Here, we provide a tutorial on how to use the LlaMA models.

This tutorial will focus on the first two steps. After these steps, the model can be trained for unsupervised or supervised contrastive learning like any other encoder model.

In this tutorial, we will transform LlaMA models into text encoders, however, transforming Mistral will require similar steps. We will focus on modifying the flash attention implementation as it requires the least changes in the codebase, and the implementation is consistent across models and transformers versions. Our tutorial is based on transformers version 4.39.3.

## 1) Enabling Bidirectional Attention

A decoder-only causal LLM consists of multiple decoder layers, each of which has a self-attention mechanism. 

<p align="center">
  <img src="https://github.com/McGill-NLP/llm2vec/blob/weblog/docs/assets/images/LLM2Vec-tutorial.png?raw=true" width="75%" alt="Llama Conceptual overview"/>
</p>

We start bottoms-up by first modifying the attention mechanism to be bidirectional.

HuggingFace implements three attention mechanisms for Llama and Mistral models - Eager, SDPA, and Flash Attention. Here, we only modify the flash attention implementation. In order to be able to use the bidirectional attention, we need to create new LLaMA flash attention class:
```python
class ModifiedLlamaFlashAttention2(LlamaFlashAttention2):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False  # Initially `True` in transformers implementation

LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": ModifiedLlamaFlashAttention2,  # Initially, `LlamaFlashAttention2'
    "sdpa": LlamaSdpaAttention,
}
```
We have changed flash attention to be non-causal (i.e., bidirectional). Next, we need to modify the decoder layer to use this new attention classes. the `__init__` function is directly copied from the `transformers` implementation of `LlamaDecoderLayer`. As `flash_attention_2` in `LLAMA_ATTENTION_CLASSES` points to the new flash attention class, the decoder layer will use bidirectional attention when initialized with `flash_attention_2`.
```python
class ModifiedLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        nn.Module.__init__(self) # Initially, super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```
Finally, we need to modify the main model class to use the new decoder layer. We create a new model class `LlamaBiModel` that inherits from `LlamaModel` and uses the new `ModifiedLlamaDecoderLayer` in its `__init__` function. Everything else remains the same as the original implementation of `LlamaModel`.
```python
class LlamaBiModel(LlamaModel):
```

We first have to use the `ModifiedLlamaDecoderLayer` in our `LlamaBiModel` class.
```python
class LlamaBiModel(LlamaModel):
    def __init__(self, config):
        LlamaPreTrainedModel.__init__(self, config) # Initially, super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [ModifiedLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]  # Initially, `LlamaDecoderLayer(config, layer_idx)`
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        self.post_init()
```
<!-- attach full file -->
That's it! We have successfully created a bidirectional LLaMA model. We can now use this model for training with masked next token prediction.


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
