#!/usr/bin/env python3

from my_transformers import load_asset, pipeline
import json


DATUMS = [
    # "Hello, I'm a language model,",
]


def main() -> None:
    # from transformers import T5ForConditionalGeneration, T5Tokenizer
    # model = T5ForConditionalGeneration.from_pretrained('t5-small')
    # tokenizer = T5Tokenizer.from_pretrained('t5-small')
    summarizer = pipeline("summarization", model="t5-large")
    for DATA in DATUMS:
        # tokens = tokenizer.encode("summarize: " + DATA, return_tensors="pt")
        # summary = model.generate(tokens,
        #                             num_beams=4,
        #                             no_repeat_ngram_size=2,
        #                             min_length=200,
        #                             max_length=300)
        for output in summarizer(DATA, max_length=300, min_length=100):
            #print(len(output))
            #result = tokenizer.decode(output, skip_special_tokens=True)
            print(json.dumps(output["summary_text"]))
            #print(result)


if __name__ == '__main__':
    main()
 
