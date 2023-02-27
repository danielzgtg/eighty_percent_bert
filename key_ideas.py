#!/usr/bin/env python3

from my_transformers import KeyBERT, to_gpu, SentenceTransformer


def _load(input_path: str) -> list[str]:
    results: list[str] = []
    result: list[str] = []
    group: str = "1"
    import csv
    with open(input_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[0] != group:
                if int(row[0]) != int(group) + 1:
                    raise ValueError("Not monotonic")
                group = row[0]
                results.append(" ".join(result))
                result = []
            result.append(row[2])
    if result:
        results.append(" ".join(result))
    return results


def _summarize(ideas: list[str]) -> None:
    kw_model = KeyBERT(to_gpu(SentenceTransformer("all-MiniLM-L6-v2")))
    for interview in ideas:
        keywords = kw_model.extract_keywords(interview)
        print(keywords)


def key_ideas(input_path: str) -> None:
    ideas: list[list[str]] = _load(input_path)
    _summarize(ideas)


def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: ./key_ideas.py ./transcript.csv")
        exit(1)
    key_ideas(sys.argv[1])


if __name__ == '__main__':
    main()
