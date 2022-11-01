#!/usr/bin/env python3
import json

from my_transformers import load_asset, pipeline


DATUMS = [
    # "Hello, I'm a language model,",
    load_asset("tokyo_completed"),
]


def main() -> None:
    classifier = pipeline("token-classification", model="mrm8488/mobilebert-finetuned-pos", device="vulkan")
    for DATA in DATUMS:
        print(json.dumps(classifier(DATA), indent=4, default=lambda x: str(x)))


if __name__ == '__main__':
    main()
