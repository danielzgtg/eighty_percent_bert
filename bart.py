#!/usr/bin/env python3

from my_transformers import load_asset, pipeline
import json


DATUMS = [
    # "Hello, I'm a language model,",
    load_asset("tokyo_completed")
]


def main() -> None:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    for DATA in DATUMS:
        for output in summarizer(DATA, max_length=130, min_length=30):
            print(json.dumps(output["summary_text"]))


if __name__ == '__main__':
    main()
