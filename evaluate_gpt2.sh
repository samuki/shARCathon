#!/usr/bin/env bash

# This script tests all possible versions

python -m gpt2 basic basic tiny
python -m gpt2 finetuned basic tiny
python -m gpt2 basic step-by-step tiny
python -m gpt2 finetuned step-by-step tiny

python -m gpt2 basic basic small
python -m gpt2 finetuned basic small
python -m gpt2 basic step-by-step small
python -m gpt2 finetuned step-by-step small

python -m gpt2 basic basic normal
python -m gpt2 finetuned basic normal
python -m gpt2 basic step-by-step normal
python -m gpt2 finetuned step-by-step normal
