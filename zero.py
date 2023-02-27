#!/usr/bin/env python3

from my_transformers import gpu_device, load_asset, pipeline


DATUMS = [
    # "Hello, I'm a language model,",
    load_asset("tokyo_completed")
]
# These candidates are categories from Wikipedia
CANDIDATES = [
    "travel",
    "cooking",
    "dancing",
    "exploration",
    "politics",
    "public health",
    "economics",
    "sports",
]


def main() -> None:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=gpu_device)
    for DATA in DATUMS:
        result = classifier(DATA, CANDIDATES)
        print(result["sequence"])
        for label, score in zip(result["labels"], result["scores"]):
            print(label, score)


if __name__ == '__main__':
    main()
