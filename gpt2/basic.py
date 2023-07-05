#!/usr/bin/env python3

from transformers import pipeline, set_seed, GPT2TokenizerFast

import sys
# Get access to the utils from one folder above
sys.path.append('../')
import utils


# For reproducability
set_seed(42)

# Configuration
# MODEL = 'gpt2'
MODEL = 'gpt2-large'
# MODEL = 'gpt2-xl'
MAX_NO_TOKENS = 1024

TOKENIZER = GPT2TokenizerFast.from_pretrained(MODEL)


def get_expected_result(data):
    res = ""
    d = data['test'][0]['output']
    in_v = minimize_list_of_list(d)
    res += "Out: " + in_v

    return res


def minimize_list_of_list(ll):
    res = ""
    for lst in ll:
        # Remove commas to reduce num of tokens required
        # and separate list of lists with ;
        res += " ".join(map(str, lst)) + "; "
    return res[:-2]


def basic_short_prompts(data):
    res_ls = []
    for d in data['train']:
        res = "Continue the pattern:"
        in_v = minimize_list_of_list(d['input'])
        res += "\nIn: " + in_v
        out_v = minimize_list_of_list(d['output'])
        res += "\nOut: " + out_v

        # Add test string
        d = data['test'][0]
        in_v = minimize_list_of_list(d['input'])
        res += "\nIn: " + in_v
        res += "\nOut: "

        res_ls.append(res)

    return res_ls


# This is typically too long (over 1024 tokens)
# Therefore we have to dynamically fallback to the method above
def basic_prompt(data):
    res = "Please continue the pattern:"
    for d in data['train']:
        in_v = minimize_list_of_list(d['input'])
        res += "\nIn: " + in_v
        out_v = minimize_list_of_list(d['output'])
        res += "\nOut: " + out_v

    # Add test string
    d = data['test'][0]
    in_v = minimize_list_of_list(d['input'])
    res += "\nIn: " + in_v
    res += "\nOut: "

    return res


def get_basic_prompts(data):
    # First try the basic approach
    basic_res = basic_prompt(data)
    answer_tokens = len(TOKENIZER(get_expected_result(data))['input_ids'])

    tokens = TOKENIZER(basic_res)
    no_tokens = len(tokens['input_ids']) + answer_tokens
    if no_tokens <= MAX_NO_TOKENS:
        return [basic_res]

    # Fall back to our next approach
    basic_short_res = basic_short_prompts(data)

    valid = True
    for r in basic_short_res:
        tokens = TOKENIZER(r)
        no_tokens = len(tokens['input_ids']) + answer_tokens
        if no_tokens > MAX_NO_TOKENS:
            valid = False
    if valid:
        return basic_short_res

    # What to do in this case?
    print("\t|> Skipping this task (required tokens > 1024)", flush=True)
    return []


def basic_approach(prompt, max_len=1024):
    generator = pipeline('text-generation', model=MODEL)
    result = generator(prompt, num_return_sequences=1, max_length=max_len)
    result = result[0]['generated_text'].removeprefix(prompt)
    return result


if __name__ == '__main__':
    logger = utils.get_logger()
    # TODO pjordan: This is for debugging only
    logger.info = print
    data = utils.load_json_data('../data/evaluation/')
    for task, value in data.items():
        logger.info(f"\t|> Task: {task}")
        prompts = get_basic_prompts(value)
        exp_result = get_expected_result(value)
        for prompt in prompts:
            logger.info(f"\t|> Prompt: \n{prompt}")
            expected_result = get_expected_result(value)
            no_tokens = len(TOKENIZER(prompt)['input_ids']) \
                + len(TOKENIZER(expected_result)['input_ids'])
            result = basic_approach(prompt, max_len=(no_tokens))
            # NOTE pjordan: We can optimize this further by not generating
            # unwanted tokens: Do something like len(prompt) + len(exp_result)
            # but over the tokens and not string length
            logger.info(f"\t|> Result: \n{result}")