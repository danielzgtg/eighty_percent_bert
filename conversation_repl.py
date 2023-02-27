#!/usr/bin/env python3

from my_transformers import gpu_device, pipeline, Conversation


def main() -> None:
    interlocutor = pipeline('conversational', model='microsoft/DialoGPT-medium', max_length=4096, device=gpu_device)
    conversation = Conversation()
    while True:
        try:
            line = input(f"Human: ")
        except (EOFError, KeyboardInterrupt):
            print("Stopping")
            return
        line = line.strip()
        if not line:
            print("Resetting conversation")
            print(repr(conversation))
            conversation = Conversation()
            continue
        conversation.add_user_input(line)
        interlocutor(conversation)
        print("Bot:", conversation.generated_responses[-1])
    print(repr(conversation))


if __name__ == '__main__':
    main()
