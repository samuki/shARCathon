#!/usr/bin/env bash

# Install pip dependencies
pip install -r ./requirements.txt


# Prepare data
unzip ./data/ARC-800-tasks.zip -d ./data
python filter_data.py
mv ./data/evaluation ./data/evaluation_orig
mv ./data/evaluation_small ./data/evaluation


# This script tests all possible versions

# # GPT 2
echo "Evaluating GPT2"
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

# GPT Neo
echo "Evaluating GPT Neo"
TOKENIZERS_PARALLELISM="false" python -m gptneo basic basic tiny
TOKENIZERS_PARALLELISM="false" python -m gptneo basic step-by-step tiny
TOKENIZERS_PARALLELISM="false" python -m gptneo basic apply-patterns tiny
# TOKENIZERS_PARALLELISM="false" python -m gptneo finetuned basic tiny
# TOKENIZERS_PARALLELISM="false" python -m gptneo finetuned step-by-step tiny
# TOKENIZERS_PARALLELISM="false" python -m gptneo finetuned apply-patterns tiny

TOKENIZERS_PARALLELISM="false" python -m gptneo basic basic small
TOKENIZERS_PARALLELISM="false" python -m gptneo basic step-by-step small
TOKENIZERS_PARALLELISM="false" python -m gptneo basic apply-patterns small
# TOKENIZERS_PARALLELISM="false" python -m gptneo finetuned basic small
# TOKENIZERS_PARALLELISM="false" python -m gptneo finetuned step-by-step small
# TOKENIZERS_PARALLELISM="false" python -m gptneo finetuned apply-patterns small

TOKENIZERS_PARALLELISM="false" python -m gptneo basic basic normal
TOKENIZERS_PARALLELISM="false" python -m gptneo basic step-by-step normal
TOKENIZERS_PARALLELISM="false" python -m gptneo basic apply-patterns normal
# TOKENIZERS_PARALLELISM="false" python -m gptneo finetuned basic normal
# TOKENIZERS_PARALLELISM="false" python -m gptneo finetuned step-by-step normal
# TOKENIZERS_PARALLELISM="false" python -m gptneo finetuned apply-patterns normal

# GPT J (20 Gig are not enough)
# echo "Evaluating GPT J"
# TOKENIZERS_PARALLELISM="false" python -m gptj basic basic tiny
# # TOKENIZERS_PARALLELISM="false" python -m gptj finetuned basic tiny
# TOKENIZERS_PARALLELISM="false" python -m gptj basic step-by-step tiny
# # TOKENIZERS_PARALLELISM="false" python -m gptj finetuned step-by-step tiny
# TOKENIZERS_PARALLELISM="false" python -m gptj basic apply-patterns tiny
# # TOKENIZERS_PARALLELISM="false" python -m gptj finetuned apply-patterns tiny
# 
# TOKENIZERS_PARALLELISM="false" python -m gptj basic basic small
# # TOKENIZERS_PARALLELISM="false" python -m gptj finetuned basic small
# TOKENIZERS_PARALLELISM="false" python -m gptj basic step-by-step small
# # TOKENIZERS_PARALLELISM="false" python -m gptj finetuned step-by-step small
# TOKENIZERS_PARALLELISM="false" python -m gptj basic apply-patterns small
# # TOKENIZERS_PARALLELISM="false" python -m gptj finetuned apply-patterns small
# 
# TOKENIZERS_PARALLELISM="false" python -m gptj basic basic normal
# # TOKENIZERS_PARALLELISM="false" python -m gptj finetuned basic normal
# TOKENIZERS_PARALLELISM="false" python -m gptj basic step-by-step normal
# # TOKENIZERS_PARALLELISM="false" python -m gptj finetuned step-by-step normal
# TOKENIZERS_PARALLELISM="false" python -m gptj basic apply-patterns normal
# # TOKENIZERS_PARALLELISM="false" python -m gptj finetuned apply-patterns normal
