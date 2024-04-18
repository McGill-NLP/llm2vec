import datasets
import numpy as np
from sklearn.metrics.pairwise import paired_cosine_distances
from scipy.stats import spearmanr

import torch
from llm2vec import LLM2Vec


dataset = "mteb/sts17-crosslingual-sts"
instruction = "Retrieve semantically similar text: "

dataset = datasets.load_dataset(dataset, "en-en")

min_score, max_score = 0, 5
normalize = lambda x: (x - min_score) / (max_score - min_score)
normalized_scores = list(map(normalize, dataset["test"]["score"]))
batch_size = 8

sentences1, sentences2 = dataset["test"]["sentence1"], dataset["test"]["sentence2"]

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

print(f"Encoding {len(sentences1)} sentences1...")
sentences1 = append_instruction(instruction, sentences1)
embeddings1 = np.asarray(model.encode(sentences1, batch_size=batch_size))

print(f"Encoding {len(sentences2)} sentences2...")
sentences2 = append_instruction(instruction, sentences2)
embeddings2 = np.asarray(model.encode(sentences2, batch_size=batch_size))

print("Evaluating...")
cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
cosine_spearman, _ = spearmanr(normalized_scores, cosine_scores)

results = {
    "cos_sim": {
        "spearman": cosine_spearman,
    }
}

print(results)
# {'cos_sim': {'spearman': 0.9021906216635642}}
