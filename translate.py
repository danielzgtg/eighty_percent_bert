#!/usr/bin/env python3

from my_transformers import load_asset, MBartForConditionalGeneration, MBart50TokenizerFast, to_gpu

DATUMS = [
    #"Hello, I'm a language model.",
    load_asset("tokyo_completed"),
]


def main() -> None:
    target_lang = "ja_XX"
    model_name = "facebook/mbart-large-50-one-to-many-mmt"
    model = to_gpu(MBartForConditionalGeneration.from_pretrained(model_name))
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="en_XX")
    for DATA in DATUMS:
        inputs = to_gpu(tokenizer(DATA, return_tensors="pt"))
        # print(len(inputs.input_ids[0]))
        # print(tokenizer.batch_decode(inputs.input_ids, skip_special_tokens=True)[0])
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
            # min_length=135,
            max_length=1024,
        )
        # print(len(translated_tokens[0]))
        print(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()
