#!/usr/bin/env python3

from my_transformers import gpu_device, pipeline


DATUMS = [
    # "Hello, I'm a language model,",
    ({
        "Day Temperature": ["31"],
        "Normal Temperature": ["24"],
        "Night Temperature": ["18"],
     }, "How cold is it at night?")
]


def main() -> None:
    answerer = pipeline("table-question-answering", model="google/tapas-base-finetuned-wtq", device=gpu_device)
    for DATA in DATUMS:
        output = answerer(DATA[0], DATA[1])
        for coords, cell in zip(output["coordinates"], output["cells"]):
            print(coords, cell)


if __name__ == '__main__':
    main()
