#!/usr/bin/env python3

from transformers import pipeline, set_seed

import sys
# Get access to the utils from one folder above
sys.path.append('../')
import utils


# For reproducability
set_seed(42)


def get_expected_result(data):
    res = ""
    d = data['test'][0]
    in_v = str(d['output']).replace(',', '')
    res += "Out: " + in_v

    return f"{data}"


def minimize_list_of_list(ll):
    res = ""
    for lst in ll:
        # Remove commas to reduce num of tokens required
        # and separate list of lists with ;
        res += " ".join(map(str, lst)) + "; "
    return res[:-2]


# This is also too large!
# 1024 tokens just aren't that many!
# TODO pjordan: Fix this and remove the [0:1] after doing so
def basic_short_prompts(data):
    res_ls = []
    for d in data['train']:
        res = "Continue the pattern:"
        in_v = minimize_list_of_list(d['input'][0:1])
        res += "\nIn: " + in_v
        out_v = minimize_list_of_list(d['output'][0:1])
        res += "\nOut: " + out_v

        # Add test string
        d = data['test'][0]
        in_v = minimize_list_of_list(d['input'][0:1])
        res += "\nIn: " + in_v
        res += "\nOut: "

        res_ls.append(res)

    return res_ls


# This is typically too long (over 1024 tokens)
# Therefore we have to dynamically fallback to the method above
# TODO pjordan: Do that
def basic_prompts(data):
    res = "Please continue the pattern:"
    for d in data['train'][0]:
        in_v = minimize_list_of_list(d['input'])
        res += "\nIn: " + in_v
        out_v = minimize_list_of_list(d['output'])
        res += "\nOut: " + out_v

    # Add test string
    d = data['test'][0]
    in_v = minimize_list_of_list(d['input'])
    res += "\nIn: " + in_v
    res += "\nOut: "

    return [res]


def basic_approach(prompt, max_len=1024):
    generator = pipeline('text-generation', model='gpt2')
    # generator = pipeline('text-generation', model='gpt2-large')
    result = generator(prompt, num_return_sequences=1, max_length=max_len)
    result = result[0]['generated_text'].removeprefix(prompt)
    return result


if __name__ == '__main__':
    logger = utils.get_logger()
    # TODO pjordan: This is for debugging only
    logger.info = print
    data = utils.load_json_data('../data/evaluation/')
    for task, value in data.items():
        logger.info(f"Task: {task}")
        prompts = basic_short_prompts(value)
        exp_result = get_expected_result(value)
        for prompt in prompts:
            logger.info(f"Prompt: {prompt}")
            result = basic_approach(prompt, max_len=(1024))
            # NOTE pjordan: We can optimize this further by not generating
            # unwanted tokens: Do something like len(prompt) + len(exp_result)
            # but over the tokens and not string length
            logger.info(f"Result: {result}")
