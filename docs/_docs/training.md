---
title: "Model Training"
permalink: /docs/training
excerpt: "Instructions to reproduce the model training in the project"
---

You can add instructions here to teach you the model training in the project.

## Load data

Add instructions on how to ingest the data, e.g.

```python
import pandas as pd

df = pd.read_csv('project_data.csv')
# ...
```

## Load model

Add instructions on how to load your model, e.g.

```python
import transformers

model = transformers.AutoModel.from_pretrained(...)
# ...
```

## Training loop

Add instructions on how to write the training loop.

### Loss function

Specify here what loss function you might be using.

### Evaluation

Specify here how to evaluate after an epoch of training.