#!/usr/bin/env python3

from my_transformers import AutoModelForCausalLM, AutoTokenizer, to_gpu
import json


DATUMS = [
    "def hello_world():",
    "# this function prints hello world",
]

MODEL_NAME = "Salesforce/codegen-2B-mono"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL = to_gpu(AutoModelForCausalLM.from_pretrained(MODEL_NAME).half())


def main() -> None:
    for DATA in DATUMS:
        input_ids = to_gpu(TOKENIZER(DATA, return_tensors="pt")).input_ids
        generated_ids = MODEL.generate(input_ids, max_length=128)
        output = TOKENIZER.decode(generated_ids[0], skip_special_tokens=True)
        print(output)


if __name__ == '__main__':
    main()
