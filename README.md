# *LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders* 
LLM2Vec is a simple recipe to convert decoder-only LLMs into text encoders. It consists of 3 simple steps: 1) enabling bidirectional attention, 2) training with masked next token prediction, and 3) unsupervised contrastive learning. The model can be further fine-tuned to achieve state-of-the-art performance.

<p align="center">
  <img src="https://github.com/McGill-NLP/llm2vec/assets/12207571/48efd48a-431b-4625-8e0f-248a442e3839" width="75%" alt="LLM2Vec_figure1"/>
</p>

## Installation
To use LLM2Vec, first install the llm2vec package from PyPI.

```bash
pip install llm2vec
```
You can also directly install it from our code by cloning the repository and: 

```bash
pip install -e .
```

## Getting Started
LLM2Vec class is a wrapper on top of HuggingFace models to support sequence encoding and pooling operations. The steps below showcase an example on how to use the library.

### Preparing the model
Here, we first initialize the model and apply MNTP-trained LoRA weights on top. After merging the model with MNTP weights, we can
- either load the unsupervised-trained LoRA weights (trained with SimCSE objective and wiki corpus)
- or we can load the model with supervised-trained LoRA weights (trained with contrastive learning and public E5 data).

```python
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel

config = AutoConfig.from_pretrained("McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp")

# Loading base MNTP model, along with custom code that enables bidirectional connections in decoder-only LLMs
tokenizer = AutoTokenizer.from_pretrained(
    "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"
)
config = AutoConfig.from_pretrained(
    "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp", trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
    trust_remote_code=True,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
)
model = PeftModel.from_pretrained(
    model,
    "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
)
model = model.merge_and_unload()  # This can take several minutes on cpu

# Loading unsupervised-trained LoRA weights. This loads the trained LoRA weights on top of MNTP model. Hence the final weights are -- Base model + MNTP (LoRA) + SimCSE (LoRA).
model = PeftModel.from_pretrained(
    model, "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-unsup-simcse"
)

# Or loading supervised-trained LoRA weights
model = PeftModel.from_pretrained(
    model, "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised"
)

```

### Applying `LLM2Vec` wrapper
Then, we define our LLM2Vec encoder model as follows:

```python
from llm2vec import LLM2Vec

l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)
```

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

# Model List

# Training 
Training code will be available soon.

# Bugs or questions?
If you have any question about the code, feel free to email Parishad (`parishad.behnamghader@mila.quebec`) and Vaibhav (`vaibhav.adlakha@mila.quebec`).
