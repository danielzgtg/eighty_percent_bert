#!/usr/bin/env python3

from my_transformers import pipeline
import json


DATUMS = [
    # "Hello, I'm a language model,",
    #load_asset("tokyo_completed")
    "Our vision is to make a Software Requirements Specification document for the Teaching Assistant Management Systems learn requirement engineering."
]


def main() -> None:
    generator = pipeline('text-generation', model='gpt2')
    for DATA in DATUMS:
        for output in generator(DATA, max_length=1000, num_return_sequences=5):
            print(json.dumps(output["generated_text"]))


if __name__ == '__main__':
    main()
