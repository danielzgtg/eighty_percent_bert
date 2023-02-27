#!/usr/bin/env python3

from my_transformers import load_asset, MBartForConditionalGeneration, MBart50TokenizerFast, to_gpu

SRC_LANG_DEFAULT: str = "en"
TGT_LANG_DEFAULT: str = "fr"
LANG_LOOKUP: dict[str, str] = {
    "ar": "ar_AR",
    "de": "de_DE",
    "en": "en_XX",
    "es": "es_XX",
    "fr": "fr_XX",
    "hi": "hi_IN",
    "it": "it_IT",
    "ja": "ja_XX",
    "ko": "ko_KR",
    "ru": "ru_RU",
    "zh": "zh_CN",
}


def find_lang(two_letter_lang: str, default_two_letter_lang: str) -> str:
    result: str | None = None
    if len(two_letter_lang) != 2:
        print(f"Language {two_letter_lang} must be two letters long")
    else:
        result = LANG_LOOKUP.get(two_letter_lang)
        if not result:
            print(f"Language {two_letter_lang} not found")
    if not result:
        print(f"Defaulting to language {default_two_letter_lang}")
        result = LANG_LOOKUP[default_two_letter_lang]
    return result


def main() -> None:
    model_name = "facebook/mbart-large-50-one-to-many-mmt"
    model = to_gpu(MBartForConditionalGeneration.from_pretrained(model_name))
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    tokenizer.src_lang = LANG_LOOKUP[SRC_LANG_DEFAULT]
    target_lang = LANG_LOOKUP[TGT_LANG_DEFAULT]
    while True:
        try:
            line = input(f"{tokenizer.src_lang}-{target_lang} > ")
        except (EOFError, KeyboardInterrupt):
            print("Stopping")
            return
        line = line.strip()
        if not line:
            continue
        if "<" in line or ">" in line:
            print("Token sanitization violation")
            continue
        # Uncomment once many-to-many is decided
        #if line.startswith("srclang_"):
        #    tokenizer.src_lang = find_lang(line[8:], SRC_LANG_DEFAULT)
        #    continue
        if line.startswith("tgtlang_"):
            target_lang = find_lang(line[8:], TGT_LANG_DEFAULT)
            continue
        inputs = to_gpu(tokenizer(line, return_tensors="pt"))
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
            # min_length=135,
            max_length=1024,
        )
        # There is a problem with the quality:
        # - Goodbye! -> Discours! ???
        # - How are you? -> Comment va-t-il? ???
        print(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()
