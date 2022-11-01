#!/usr/bin/env python3

from my_transformers import pipeline, Conversation


DATUMS = [
    "Hello",
    "What do you think about Tokyo?",
    "Me too! How were they like when you last met them?",
]


def main() -> None:
    interlocutor = pipeline('conversational', model='microsoft/DialoGPT-medium', max_length=4096)
    conversation = Conversation()
    for DATA in DATUMS:
        conversation.add_user_input(DATA)
        interlocutor(conversation)
    print(repr(conversation))


if __name__ == '__main__':
    main()
