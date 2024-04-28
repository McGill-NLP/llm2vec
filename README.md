# *LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders* 

[![arxiv](https://img.shields.io/badge/arXiv-2404.05961-b31b1b.svg)](https://arxiv.org/abs/2404.05961)
[![PyPi](https://img.shields.io/pypi/v/llm2vec)](https://pypi.org/project/llm2vec/)
[![HF](https://img.shields.io/badge/HF%20Models-LLM2Vec-FFD21E.svg)](https://huggingface.co/collections/McGill-NLP/llm2vec-660e14f536b3c8d10b3f1c34)




LLM2Vec is a simple recipe to convert decoder-only LLMs into text encoders. It consists of 3 simple steps: 1) enabling bidirectional attention, 2) training with masked next token prediction, and 3) unsupervised contrastive learning. The model can be further fine-tuned to achieve state-of-the-art performance.

<p align="center">
  <img src="https://github.com/McGill-NLP/llm2vec/assets/12207571/48efd48a-431b-4625-8e0f-248a442e3839" width="75%" alt="LLM2Vec_figure1"/>
</p>

## Installation
To use LLM2Vec, first install the llm2vec package from PyPI, followed by installing flash-attention:

```bash
pip install llm2vec
pip install flash-attn --no-build-isolation
```
You can also directly install the latest version of llm2vec by cloning the repository: 

```bash
pip install -e .
pip install flash-attn --no-build-isolation
```

## Getting Started
LLM2Vec class is a wrapper on top of HuggingFace models to support enabling bidirectionality in decoder-only LLMs, sequence encoding and pooling operations. The steps below showcase an example on how to use the library.

### Preparing the model
Initializing LLM2Vec model using pretrained LLMs is straightforward. The `from_pretrained` method of LLM2Vec takes a base model identifier/path and an optional PEFT model identifier/path. All HuggingFace model loading arguments can be passed to `from_pretrained` method. By default, the models are loaded with bidirectional connections enabled. This can be turned off by passing `enable_bidirectional=False` to the `from_pretrained` method.

Here, we first initialize the Mistral MNTP base model and load the unsupervised-trained LoRA weights (trained with SimCSE objective and wiki corpus).

```python
import torch
from llm2vec import LLM2Vec

l2v = LLM2Vec.from_pretrained(
    "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
    peft_model_name_or_path="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
)
```

We can also load the model with supervised-trained LoRA weights (trained with contrastive learning and public E5 data) by changing the `peft_model_name_or_path`.

```python
import torch
from llm2vec import LLM2Vec

l2v = LLM2Vec.from_pretrained(
    "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
    peft_model_name_or_path="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
)
```

By default the LLM2Vec model uses the `mean` pooling strategy. You can change the pooling strategy by passing the `pooling_mode` argument to the `from_pretrained` method. Similarly, you can change the maximum sequence length by passing the `max_length` argument (default is 512).

### Inference
This model now returns the text embedding for any input in the form of `[[instruction1, text1], [instruction2, text2]]` or `[text1, text2]`. While training, we provide instructions for both sentences in symmetric tasks, and only for for queries in asymmetric tasks.

```python
# Encoding queries using instructions
instruction = (
    "Given a web search query, retrieve relevant passages that answer the query:"
)
queries = [
    [instruction, "how much protein should a female eat"],
    [instruction, "summit define"],
]
q_reps = l2v.encode(queries)

# Encoding documents. Instruction are not required for documents
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
]
d_reps = l2v.encode(documents)

# Compute cosine similarity
q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
d_reps_norm = torch.nn.functional.normalize(d_reps, p=2, dim=1)
cos_sim = torch.mm(q_reps_norm, d_reps_norm.transpose(0, 1))

print(cos_sim)
"""
tensor([[0.5485, 0.0551],
        [0.0565, 0.5425]])
"""
```

More examples of classification, clustering, sentence similarity etc are present in [examples](examples) directory.

## Model List
- ### Mistral-7B
  - ### [Bi + MNTP](https://huggingface.co/McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp)
  - ### [Bi + MNTP + SimCSE](https://huggingface.co/McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-unsup-simcse) (Unsupervised state-of-the-art on MTEB)
  - ### [Bi + MNTP + Supervised](https://huggingface.co/McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised) (state-of-the-art on MTEB among models trained on public data)
- ### Llama-2-7B
  - ### [Bi + MNTP](https://huggingface.co/McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp)
  - ### [Bi + MNTP + SimCSE](https://huggingface.co/McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-unsup-simcse)
  - ### [Bi + MNTP + Supervised](https://huggingface.co/McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-supervised)
- ### Sheared-Llama-1.3B
  - ### [Bi + MNTP](https://huggingface.co/McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp)
  - ### [Bi + MNTP + SimCSE](https://huggingface.co/McGill-NLP/LLM2Vec-Sheared-LLaMA-unsup-simcse)
  - ### [Bi + MNTP + Supervised](https://huggingface.co/McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised)

## Training 
### MNTP training
To train the model with Masked Next Token Prediction (MNTP), you can use the `experiments/run_mntp.py` script. It is adapted from HuggingFace Masked Language Modeling (MLM) [script](https://github.com/huggingface/transformers/blob/51bcadc10a569847b93a30dbe3a077037ae63bad/examples/pytorch/language-modeling/run_mlm.py). To train the Mistral-7B model with MNTP, run the following command:

```bash
python experiments/run_mntp.py train_configs/mntp/Mistral.json
```

The Mistral training configuration [file](train_configs/mntp/Mistral.json) contains all the training hyperparameters and configurations used in our paper. 
```json
{
    "model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.2",
    "dataset_name": "wikitext",
    "dataset_config_name": "wikitext-103-raw-v1",
    "mask_token_type": "blank",
    "data_collator_type": "all_mask",
    "mlm_probability": 0.8,
    "lora_r": 16,
    "gradient_checkpointing": true,
    "torch_dtype": "bfloat16",
    "attn_implementation": "flash_attention_2"
    // ....
}
```

Similar configurations are also available for [Llama-2-7B](train_configs/mntp/Llama.json) and [Sheared-Llama-1.3B](train_configs/mntp/Sheared-Llama.json) models.

## Citation
If you find our work helpful, please cite us:
```bibtex
@article{llm2vec,
      title={{LLM2Vec}: {L}arge Language Models Are Secretly Powerful Text Encoders}, 
      author={Parishad BehnamGhader and Vaibhav Adlakha and Marius Mosbach and Dzmitry Bahdanau and Nicolas Chapados and Siva Reddy},
      year={2024},
      journal={arXiv preprint},
      url={https://arxiv.org/abs/2404.05961}
}
```

## Bugs or questions?
If you have any questions about the code, feel free to open an issue on the GitHub repository.
