#!/usr/bin/env python3

from transformers import pipeline
import numpy as np

from . import \
    DEBUG, MODEL, NO_GENERATED_RESULTS, \
    MAX_NO_TOKENS, TOKENIZER, DATA_DIR, \
    load_json_data, get_expected_result, \
    get_logger
from .prompts import get_prompts


def train():
    pass


EXPECTED_CHARS = [str(i) for i in range(0, 10)] + [';', ' ']


def select_best_answer(answers):
    if len(answers) <= 0:
        return None
    if len(answers) == 1:
        return answers[0]

    # Simple approach: Pick the answer with the least 'non-expected chars'
    scores = len(answers) * [0]
    for i, answer in enumerate(answers):
        score = 0
        for c in answer:
            if c not in EXPECTED_CHARS:
                score += 1
        scores[i] = score
    return answers[np.argmin(scores)]


def basic_generator(prompt, max_len=MAX_NO_TOKENS):
    # TODO pjordan: Use finetuned model instead
    generator = pipeline('text-generation', model=MODEL)
    results = generator(
        prompt, num_return_sequences=NO_GENERATED_RESULTS, max_length=max_len)
    answers = [result['generated_text'].removeprefix(
        prompt) for result in results]
    result = select_best_answer(answers)
    return result


def main(kind='basic'):
    logger = get_logger()
    if DEBUG:
        logger.info = print
    data = load_json_data(DATA_DIR)
    for task, value in data.items():
        logger.info(f"\t|> Task: {task}")
        prompts = get_prompts(value, kind=kind)
        exp_result = get_expected_result(value)
        for prompt in prompts:
            logger.info(f"\t|> Prompt: \n{prompt}")
            no_tokens = len(TOKENIZER(prompt)['input_ids']) \
                + len(TOKENIZER(exp_result)['input_ids'])
            result = basic_generator(prompt, max_len=(no_tokens))
            logger.info(f"\t|> Result: \n{result}")
