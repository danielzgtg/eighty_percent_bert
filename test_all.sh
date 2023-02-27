#!/bin/bash
# no set -e
. venv/bin/activate
./bart.py
./bert.py
./conversation.py
./distilbert-answer.py
./distilbert-classify.py
./electra.py
./gpt2.py
./multiple.py
./roberta.py
./table.py
./translate.py
./zero.py
