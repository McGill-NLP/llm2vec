import sklearn
import sklearn.cluster

import datasets
import tqdm
import numpy as np

import torch
from llm2vec import LLM2Vec

dataset = "mteb/twentynewsgroups-clustering"
instruction = "Identify the topic or theme of the given news articles: "

dataset = datasets.load_dataset(dataset)
batch_size = 32

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

v_measures = []
for cluster_set in tqdm.tqdm(dataset["test"], desc="Clustering"):
    sentences = cluster_set["sentences"]
    labels = cluster_set["labels"]
    clustering_batch_size = 500

    print(f"Encoding {len(sentences)} sentences...")
    new_sentences = append_instruction(instruction, sentences)
    corpus_embeddings = np.asarray(model.encode(new_sentences, batch_size=batch_size))

    print("Fitting Mini-Batch K-Means model...")
    clustering_model = sklearn.cluster.MiniBatchKMeans(
        n_clusters=len(set(labels)), batch_size=clustering_batch_size
    )
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    print("Evaluating...")
    v_measure = sklearn.metrics.cluster.v_measure_score(labels, cluster_assignment)
    v_measures.append(v_measure)

v_mean = np.mean(v_measures)
v_std = np.std(v_measures)

print(v_mean)
# 0.5137461051538426