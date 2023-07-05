#!/usr/bin/env python3

from . import MAX_NO_TOKENS, TOKENIZER, \
    minimize_list_of_list, get_expected_result

PROMPT_KINDS = ['basic', 'step-by-step']

PROMPT_START = {
    'basic': "Continue the pattern",
    'step-by-step': "Do the following:\nWhat is the step by step description of the input/output relation that holds for all example input/output pairs?"
}
PROMPT_DIVIDER = {
    'basic': "",
    'step-by-step': "\nYou now have all the information to solve the task. Apply this description to the following test input and write you answer as 'Out: '"
}
PROMPT_AFTER = {
    'basic': "\nOut: ",
    'step-by-step': ""
}


def short_prompts(data, kind='basic'):
    res_ls = []
    for d in data['train']:
        res = PROMPT_START[kind]
        in_v = minimize_list_of_list(d['input'])
        res += "\nIn: " + in_v
        out_v = minimize_list_of_list(d['output'])
        res += "\nOut: " + out_v

        # Add test string
        res += PROMPT_DIVIDER[kind]
        d = data['test'][0]
        in_v = minimize_list_of_list(d['input'])
        res += "\nIn: " + in_v
        res += PROMPT_AFTER[kind]

        res_ls.append(res)

    return res_ls


# This is typically too long (over 1024 tokens)
# Therefore we have to dynamically fallback to the method above
def full_prompt(data, kind='basic'):
    res = PROMPT_START[kind]

    for d in data['train']:
        in_v = minimize_list_of_list(d['input'])
        res += "\nIn: " + in_v
        out_v = minimize_list_of_list(d['output'])
        res += "\nOut: " + out_v

    # Add test string
    res += PROMPT_DIVIDER[kind]
    d = data['test'][0]
    in_v = minimize_list_of_list(d['input'])
    res += "\nIn: " + in_v
    res += PROMPT_AFTER[kind]

    return res


def get_prompts(data, kind='basic'):
    # First try the basic approach
    basic_res = full_prompt(data, kind=kind)
    answer_tokens = len(TOKENIZER(get_expected_result(data))['input_ids'])

    tokens = TOKENIZER(basic_res)
    no_tokens = len(tokens['input_ids']) + answer_tokens
    if no_tokens <= MAX_NO_TOKENS:
        return [basic_res]

    # Fall back to our next approach
    basic_short_res = short_prompts(data, kind=kind)

    valid = True
    for r in basic_short_res:
        tokens = TOKENIZER(r)
        no_tokens = len(tokens['input_ids']) + answer_tokens
        if no_tokens > MAX_NO_TOKENS:
            valid = False
    if valid:
        return basic_short_res

    # What to do in this case?
    print(f"\t|> Skipping this task (required tokens > {MAX_NO_TOKENS})",
          flush=True)
    return []
