from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
import datasets
import numpy as np

import torch
from llm2vec import LLM2Vec

dataset = "mteb/amazon_counterfactual"
instruction = "Classify a given Amazon customer review text as either counterfactual or notcounterfactual: "

dataset = datasets.load_dataset(dataset, "en")

sentences_train, y_train = dataset["train"]["text"], dataset["train"]["label"]
sentences_test, y_test = dataset["test"]["text"], dataset["test"]["label"]
max_iter = 100
batch_size = 8

scores = {}
clf = LogisticRegression(
    random_state=42,
    n_jobs=1,
    max_iter=max_iter,
    verbose=0,
)

print("Loading model...")
model = LLM2Vec.from_pretrained(
    "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
    peft_model_name_or_path="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
)

def append_instruction(instruction, sentences):
    new_sentences = []
    for s in sentences:
        new_sentences.append([instruction, s, 0])
    return new_sentences

print(f"Encoding {len(sentences_train)} training sentences...")
sentences_train = append_instruction(instruction, sentences_train)
X_train = np.asarray(model.encode(sentences_train, batch_size=batch_size))

print(f"Encoding {len(sentences_test)} test sentences...")
sentences_test = append_instruction(instruction, sentences_test)
X_test = np.asarray(model.encode(sentences_test, batch_size=batch_size))

print("Fitting logistic regression classifier...")
clf.fit(X_train, y_train)
print("Evaluating...")
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
scores["accuracy"] = accuracy
f1 = f1_score(y_test, y_pred, average="macro")
scores["f1"] = f1

print(scores)
# {'accuracy': 0.891044776119403, 'f1': 0.8283106625713033}