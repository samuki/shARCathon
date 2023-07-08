#!/usr/bin/env bash

# This script tests all possible versions

TOKENIZERS_PARALLELISM="false" python -m gpt2 basic basic tiny
TOKENIZERS_PARALLELISM="false" python -m gpt2 finetuned basic tiny
TOKENIZERS_PARALLELISM="false" python -m gpt2 basic step-by-step tiny
TOKENIZERS_PARALLELISM="false" python -m gpt2 finetuned step-by-step tiny
TOKENIZERS_PARALLELISM="false" python -m gpt2 basic apply-patterns tiny
TOKENIZERS_PARALLELISM="false" python -m gpt2 finetuned apply-patterns tiny

TOKENIZERS_PARALLELISM="false" python -m gpt2 basic basic small
TOKENIZERS_PARALLELISM="false" python -m gpt2 finetuned basic small
TOKENIZERS_PARALLELISM="false" python -m gpt2 basic step-by-step small
TOKENIZERS_PARALLELISM="false" python -m gpt2 finetuned step-by-step small
TOKENIZERS_PARALLELISM="false" python -m gpt2 basic apply-patterns small
TOKENIZERS_PARALLELISM="false" python -m gpt2 finetuned apply-patterns small

TOKENIZERS_PARALLELISM="false" python -m gpt2 basic basic normal
TOKENIZERS_PARALLELISM="false" python -m gpt2 finetuned basic normal
TOKENIZERS_PARALLELISM="false" python -m gpt2 basic step-by-step normal
TOKENIZERS_PARALLELISM="false" python -m gpt2 finetuned step-by-step normal
TOKENIZERS_PARALLELISM="false" python -m gpt2 basic apply-patterns normal
TOKENIZERS_PARALLELISM="false" python -m gpt2 finetuned apply-patterns normal
