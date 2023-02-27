#!/usr/bin/env python3

from my_transformers import gpu_device, load_asset, pipeline

DATUMS = [
    load_asset("tokyo_mask"),
    load_asset("shower_mask"),
    load_asset("help_mask"),
]


def unmask_all(unmasker, data: str) -> tuple[float, str]:
    running_score = 1
    while True:
        result = unmasker(data)
        first = result[0]
        if type(first) == dict:
            return running_score * first["score"], first["sequence"]
        first = result[0][0]
        data = first["sequence"]
        score = first["score"]
        for x in result:
            for output in x:
                new_score = output["score"]
                if new_score > score:
                    score = new_score
                    data = output["sequence"]
        data = data[6:-6]
        running_score *= score


def main() -> None:
    unmasker = pipeline('fill-mask', model='bert-base-uncased', device=gpu_device)
    for DATA in DATUMS:
        # print(json.dumps(unmasker(DATA), indent=4))
        print(*unmask_all(unmasker, DATA))


if __name__ == "__main__":
    main()

