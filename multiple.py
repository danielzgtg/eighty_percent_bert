#!/usr/bin/env python3

from my_transformers import pipeline

DATUMS = [
    (load_asset("tokyo_mask")[:115] + load_asset("tokyo_completed")[107:], ["happy", "alone", "bored"]),
    (load_asset("shower_mask")[:153], ["home", "outside", "inside"]),
    # The third-party-sourced content below is too short to be original so it is not copyrightable
    # These 3 are probably from elementary school worksheets on Google Images
    ("we are at the [MASK].", ["hat", "mat", "that", "park"]),
    ("I will [MASK] to the park.", ["my", "go", "eat"]),
    ("I see a cat [MASK] dog.", ["in", "go", "and"]),
    # From an article saying language transformers can't do math
    ("42 43 [MASK] 45 46.", ["46", "44"]),
]


def main() -> None:
    unmasker = pipeline('fill-mask', model='bert-base-uncased')
    for DATA in DATUMS:
        # print(json.dumps(unmasker(DATA), indent=4))
        print(DATA[0])
        for output in unmasker(DATA[0], targets=DATA[1]):
            print(output["token_str"], output["score"])


if __name__ == "__main__":
    main()

