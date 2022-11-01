#!/usr/bin/env python3

from my_transformers import load_asset, pipeline
import json


DATUMS = [
    # "Hello, I'm a language model,",
    load_asset("tokyo_completed"),
    load_asset("help_completed"),
]


def main() -> None:
    classifier = pipeline("token-classification", model="dbmdz/electra-large-discriminator-finetuned-conll03-english")
    for DATA in DATUMS:
        print(json.dumps(classifier(DATA), indent=4, default=lambda x: str(x)))


if __name__ == '__main__':
    main()
