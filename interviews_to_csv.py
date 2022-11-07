#!/usr/bin/env python3
import csv
import itertools
import re


PATTERN: re.Pattern = re.compile(r"[.?]+ *")
CONJUNCTIONS: re.Pattern = re.compile(r" *(?<!\w)(and|or|but|um|uh|oh|okay|let's|yes|no|right)(?!\w) *")
CONJUNCTIONS2: re.Pattern = re.compile(r" *(?<!\w)(if|then|so|because|about)(?!\w) *")


class SentenceCollector:
    def __init__(self, result: list[tuple[int, int, str]]):
        self._result: list[tuple[int, int, str]] = result
        self._seen: set[str] = set()

    def offer(self, team: int, line: int, text: str):
        if not text:
            raise ValueError("Empty line")
        if text[0] == " " or text[-1] == " ":
            raise ValueError("Text is not stripped")
        if len(text) < 12:
            # "Stopword" filter
            return
        if text in self._seen:
            print("duplicate", text)
            return
        self._seen.add(text)
        self._result.append((team, line + 1, text))


def load_team(i: int, result: SentenceCollector) -> None:
    with open(f"Round 1 - Team {i}.mp4.txt") as f:
        data: str = f.read()
    if not data:
        raise EOFError(f"Missing data for team {i}")
    buffer: str = ""
    buffer_line: int = 0
    run_on: bool = False
    for line, text in enumerate(data.splitlines()):
        split: list[str] = PATTERN.split(text)
        if buffer:
            if run_on or len(buffer) > 500:  # Emergency
                if not run_on:
                    print(f"Run on team {i}")
                    run_on = True
                while len(buffer) > 200:
                    match: re.Match | None = CONJUNCTIONS.search(buffer)
                    if not match:
                        match = CONJUNCTIONS2.search(buffer)
                    if not match:
                        raise ValueError(f"Run on sentence in team {i}")
                    if match.start():
                        result.offer(i, buffer_line, buffer[:match.start()])
                    buffer = buffer[match.end():]
                buffer_line = line  # This isn't accurate but it's better than leaving it there
        if buffer:
            buffer = f"{buffer} {split[0]}"
        else:
            buffer = split[0]
            buffer_line = line
        if len(split) < 2:
            continue
        result.offer(i, buffer_line, buffer)
        for part in itertools.islice(split, 1, len(split) - 1):
            result.offer(i, line, part)
        buffer_line = line
        buffer = split[-1]
    if buffer:
        result.offer(i, buffer_line, buffer)


def output(result: list[tuple[int, int, str]]) -> None:
    with open("interviews.csv", "w") as f:
        out = csv.writer(f)
        out.writerow(("Team", "Line", "Text"))
        out.writerows(result)


def main() -> None:
    result: list[tuple[int, int, str]] = []
    collector: SentenceCollector = SentenceCollector(result)
    for i in range(1):
        load_team(i + 1, collector)
    output(result)


if __name__ == '__main__':
    main()
