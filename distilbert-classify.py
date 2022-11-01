#!/usr/bin/env python3

from my_transformers import load_asset, pipeline


DATUMS = [
    # "Hello, I'm a language model,",
    load_asset("tokyo_completed")
]


def main() -> None:
    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    for DATA in DATUMS:
        print(DATA)
        for x in classifier(DATA, return_all_scores=True):
            for output in x:
                print(output["label"], output["score"])


if __name__ == '__main__':
    main()
