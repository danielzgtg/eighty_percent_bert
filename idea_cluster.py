#!/usr/bin/env python3

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import dendrogram
from my_transformers import AutoModel, AutoTokenizer, plt, to_gpu


__all__ = ["idea_cluster"]


plt.rcParams['svg.fonttype'] = 'none'
MODEL_NAME = "bert-base-uncased"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=40)
MODEL = to_gpu(AutoModel.from_pretrained(MODEL_NAME))


def _load(input_path: str) -> list[list[str]]:
    results: list[list[str]] = []
    result: list[str] = []
    import csv
    with open(input_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            result.append(row[2])
            if len(result) == 50:
                results.append(result)
                result = []
    if result:
        results.append(result)
    return results


def _embed(ideas: list[list[str]]) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    batches: int = len(ideas)
    for i, idea in enumerate(ideas):
        tokens = to_gpu(TOKENIZER(idea, padding=True, truncation=True, return_tensors="pt"))
        outputs = MODEL(**tokens)
        features = outputs.pooler_output.cpu().detach().numpy()
        result.extend(features[i] for i in range(len(features)))
        print(f"Batch {i}/{batches}")
    return result


def _cluster(features: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    ward = AgglomerativeClustering(compute_distances=True)
    ward.fit(features)
    return ward.children_, ward.distances_


def _present(children: np.ndarray, distances: np.ndarray, labels: list[str], output_path: str) -> None:
    linkage_matrix = np.column_stack([children, distances, np.zeros(children.shape[0])]).astype(float)
    # noinspection PyTypeChecker
    dendrogram(linkage_matrix, orientation="left", labels=labels)
    plt.gcf().set_size_inches(20, len(labels) / 10)
    plt.subplots_adjust(top=1, bottom=0, left=0, right=0.5)
    plt.savefig(output_path)


def idea_cluster(input_path: str, output_path: str) -> None:
    ideas: list[list[str]] = _load(input_path)
    embeddings: list[np.ndarray] = _embed(ideas)
    children, distances = _cluster(embeddings)
    ideas_flat: list[str] = [y for x in ideas for y in x]
    _present(children, distances, ideas_flat, output_path)


def main():
    import sys
    if len(sys.argv) != 3:
        print("Usage: ./idea_cluster.py ./transcript.csv output.svg")
        exit(1)
    idea_cluster(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()
