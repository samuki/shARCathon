#!/usr/bin/env python3

from transformers import pipeline

from . import \
    MODEL, NO_GENERATED_RESULTS, \
    MAX_NO_TOKENS, TOKENIZER, DATA_DIR, \
    load_json_data, get_expected_result, \
    select_best_answer
from .prompts import get_prompts
from .output import add_datapoint, dump_data


def basic_generator(generator, prompt, list_kind='small', max_len=MAX_NO_TOKENS):
    results = generator(
        prompt, num_return_sequences=NO_GENERATED_RESULTS, max_length=max_len)
    answers = [
        result['generated_text'].replace(prompt, '', 1)
        for result in results
    ]
    result = select_best_answer(answers, list_kind)
    return result


def main(json_path, kind='basic', list_kind='small'):
    data = load_json_data(DATA_DIR)
    generator = pipeline('text-generation', model=MODEL, device="cuda:0")
    for task, value in data.items():
        print(f"\t|> Task: {task}")
        prompts = get_prompts(value, kind=kind, list_kind=list_kind)
        exp_result = get_expected_result(value, list_kind=list_kind)
        for prompt in prompts:
            print(f"\t|> Prompt: \n{prompt}")
            print(f"\t|> Expected Result: \n{exp_result}")
            result = basic_generator(generator, prompt, list_kind=list_kind)
            print(f"\t|> Result: \n{result}")
            add_datapoint(prompt, result, exp_result, task)
    dump_data(json_path)
