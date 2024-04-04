# LLM2Vec

*LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders*. 

<p align="center">
  <img src="https://github.com/McGill-NLP/llm2vec/assets/12207571/48efd48a-431b-4625-8e0f-248a442e3839" width="75%" alt="LLM2Vec_figure1"/>
</p>

## Instrallation
To use LLM2Vec, first install the llm2vec package from PyPI.

```bash
pip install llm2vec
```
You can also directly install it from our code by cloning the repository and: 

```bash
pip install -e .
```

## Getting Started
LLM2Vec is a generic model, which takes a `tokenizer` and a `model`. First, we define the model and tokenizer using `transformers` library:

```python
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModel, AutoConfig

config = AutoConfig.from_pretrained("McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp")

model = AutoModel.from_pretrained("McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp", trust_remote_code=True, config=config, torch_dtype=torch.bfloat16)
# Loading MNTP-trained LoRA weights
model = PeftModel.from_pretrained(model, "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp")
model = model.merge_and_unload()

# Either loading unsupervised-trained LoRA weights
model = PeftModel.from_pretrained(model, "McGill-NLP/LLM2Vec-Sheared-LLaMA-unsup-simcse-mean")

# Or loading supervised-trained LoRA weights
model = PeftModel.from_pretrained(model, "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised")
```
Here, we first initialize the model and apply MNTP-trained LoRA weights on top. After merging the model with MNTP weights, we can
- either load the unsupervised-trained LoRA weights (trained with SimCSE objective and wiki corpus)
- or we can load the model with supervised-trained LoRA weights (trained with contrastive learning and public E5 data).

Then, we define our llm2vec model as follows:

```python
from llm2vec import LLM2Vec

l2v = LLM2Vec(model, tokenizer)
```

This model now returns the text embedding for any input in the form of `[[instruction, text]]`.

```python
inputs = [
  ['Retrieve duplicate questions from StackOverflow forum', 'Python (Numpy) array sorting'],
  ['', 'Sort a list in python'],
  ['', 'Sort an array in Java'],
]
repr = l2v.encode(inputs, convert_to_tensor=True)
sim_pos = torch.nn.functional.cosine_similarity(repr[0].unsqueeze(0), repr[1].unsqueeze(0))
sim_neg = torch.nn.functional.cosine_similarity(repr[0].unsqueeze(0), repr[2].unsqueeze(0))
```

# Model List

# Training 
Training code will be available soon.

# Bugs or questions?
If you have any question about the code, feel free to email Parishad (`parishad.behnamghader@mila.quebec`) and Vaibhav (`vaibhav.adlakha@mila.quebec`).
