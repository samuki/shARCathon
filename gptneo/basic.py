#!/usr/bin/env python3

from transformers import pipeline

from . import \
    MODEL, NO_GENERATED_RESULTS, TOKENIZER, DATA_DIR, \
    load_json_data, get_expected_result, \
    select_best_answer
from .prompts import get_prompts


def basic_generator(prompt, list_kind='small', max_len=None):
    generator = pipeline('text-generation', model=MODEL, device="cuda:0")
    results = generator(
        prompt, num_return_sequences=NO_GENERATED_RESULTS, max_length=max_len)
    answers = [
        result['generated_text'].replace(prompt, '', 1)
        for result in results
    ]
    result = select_best_answer(answers, list_kind)
    return result


def main(logger, kind='basic', list_kind='small'):
    data = load_json_data(DATA_DIR)
    for task, value in data.items():
        logger.info(f"\t|> Task: {task}")
        prompts = get_prompts(value, kind=kind, list_kind=list_kind)
        exp_result = get_expected_result(value, list_kind=list_kind)
        for prompt in prompts:
            logger.info(f"\t|> Prompt: \n{prompt}")
            no_tokens = len(TOKENIZER(prompt)['input_ids']) \
                + len(TOKENIZER(exp_result)['input_ids'])
            logger.info(f"\t|> Expected Result: \n{exp_result}")
            result = basic_generator(prompt, list_kind=list_kind, max_len=no_tokens)
            logger.info(f"\t|> Result: \n{result}")
