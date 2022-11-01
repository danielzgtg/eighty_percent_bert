#!/usr/bin/env python3

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from my_transformers import AutoModel, AutoTokenizer


__all__ = ["idea_cluster"]


plt.rcParams['svg.fonttype'] = 'none'
MODEL_NAME = "bert-base-uncased"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=40)
MODEL = AutoModel.from_pretrained(MODEL_NAME)


def _load(input_path: str) -> list[str]:
    result: list[str] = []
    import csv
    with open(input_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            result.append(row[2])
    return result


def _embed(ideas: list[str]) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for idea in ideas:
        tokens = TOKENIZER(idea, padding=True, truncation=True, return_tensors="pt")
        outputs = MODEL(**tokens)
        result.append(outputs.pooler_output[0].cpu().detach().numpy())
    return result


def _cluster(features: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    ward = AgglomerativeClustering(compute_distances=True)
    ward.fit(features)
    return ward.children_, ward.distances_


def _present(children: np.ndarray, distances: np.ndarray, labels, output_path: str) -> None:
    linkage_matrix = np.column_stack([children, distances, np.zeros(children.shape[0])]).astype(float)
    dendrogram(linkage_matrix, orientation="left", labels=labels)
    plt.gcf().set_size_inches(20, 50)
    plt.subplots_adjust(top=1, bottom=0, left=0, right=0.5)
    plt.savefig(output_path)


def idea_cluster(input_path: str, output_path: str) -> None:
    ideas: list[str] = _load()
    embeddings: list[np.ndarray] = _embed(ideas)
    children, distances = _cluster(embeddings)
    _present(children, distances, ideas, output_path)
