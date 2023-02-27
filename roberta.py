#!/usr/bin/env python3

from my_transformers import gpu_device, load_asset, pipeline


DATUMS = [
    "Hello, I'm a language model,",
    "The dog was lost. Nobody lost any animal",
    load_asset("tokyo_completed"),
]


def main() -> None:
    classifier = pipeline("text-classification", model="roberta-large-mnli", device=gpu_device)
    for DATA in DATUMS:
        print(DATA)
        for x in classifier(DATA, return_all_scores=True):
            for output in x:
                print(output["label"], output["score"])


if __name__ == '__main__':
    main()
