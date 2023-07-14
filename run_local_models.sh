#!/usr/bin/env bash

# This script tests all possible working versions

# GPT 2
echo "Evaluating GPT2"
python -m gpt2 basic basic no-space
python -m gpt2 finetuned basic no-space
python -m gpt2 basic step-by-step no-space
python -m gpt2 finetuned step-by-step no-space
python -m gpt2 basic apply-patterns no-space
python -m gpt2 finetuned apply-patterns no-space

python -m gpt2 basic basic tiny
python -m gpt2 finetuned basic tiny
python -m gpt2 basic step-by-step tiny
python -m gpt2 finetuned step-by-step tiny
python -m gpt2 basic apply-patterns tiny
python -m gpt2 finetuned apply-patterns tiny

python -m gpt2 basic basic small
python -m gpt2 finetuned basic small
python -m gpt2 basic step-by-step small
python -m gpt2 finetuned step-by-step small
python -m gpt2 basic apply-patterns small
python -m gpt2 finetuned apply-patterns small

python -m gpt2 basic basic normal
python -m gpt2 finetuned basic normal
python -m gpt2 basic step-by-step normal
python -m gpt2 finetuned step-by-step normal
python -m gpt2 basic apply-patterns normal
python -m gpt2 finetuned apply-patterns normal


# GPT Neo
echo "Evaluating GPT Neo"
python -m gptneo basic basic tiny
python -m gptneo basic step-by-step tiny
python -m gptneo basic apply-patterns tiny
python -m gptneo finetuned basic tiny
python -m gptneo finetuned step-by-step tiny
python -m gptneo finetuned apply-patterns tiny

python -m gptneo basic basic small
python -m gptneo basic step-by-step small
python -m gptneo basic apply-patterns small
python -m gptneo finetuned basic small
python -m gptneo finetuned step-by-step small
python -m gptneo finetuned apply-patterns small

python -m gptneo basic basic normal
python -m gptneo basic step-by-step normal
python -m gptneo basic apply-patterns normal
python -m gptneo finetuned basic normal
python -m gptneo finetuned step-by-step normal
python -m gptneo finetuned apply-patterns normal


# GPT J (20 Gig are not enough)
echo "Evaluating GPT J"
python -m gptj basic basic tiny
python -m gptj basic step-by-step tiny
python -m gptj basic apply-patterns tiny

python -m gptj basic basic small
python -m gptj basic step-by-step small
python -m gptj basic apply-patterns small

python -m gptj basic basic normal
python -m gptj basic step-by-step normal
python -m gptj basic apply-patterns normal

