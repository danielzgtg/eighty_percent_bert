#!/usr/bin/env python3

from my_transformers import gpu_device, pipeline
import json


DATUMS = [
    # "Hello, I'm a language model,",
    #(load_asset("tokyo_completed"), "Where are you?")
    # The following was from an in-person conversation with Sean D'Souza:
    ("One day a Chinese boy who is an orphan saw a heavenly scroll. "
        "This scroll contained the secrets of internal martial arts. "
        "With this scroll the boy could strike down any individual with a bolt of lightning. "
        "He sees a feng-shui master who thinks that he is scammer, "
        "but in reality the feng-shui master is a scammer. "
        "The feng-shui master has been exploiting others by asking for money in return for reading feng-shui, "
        "but in reality he doesn't actually understand feng-shui. "
        "The boy decided that justice must be served and struck down the scammer with lightning.",
        # Somehow "How did he learn internal martial arts?" fails and is <10% confidence
        "What method was used to learn internal martial arts?") # ~25% confidence
]


def main() -> None:
    answerer = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", device=gpu_device)
    for DATA in DATUMS:
        # noinspection PyArgumentList
        print(json.dumps(answerer(question=DATA[1], context=DATA[0])))


if __name__ == '__main__':
    main()
