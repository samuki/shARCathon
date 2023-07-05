#!/usr/bin/env python3

from . import MAX_NO_TOKENS, TOKENIZER, \
    minimize_list_of_list, get_expected_result


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
    res = "Continue the pattern:"
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
    print(f"\t|> Skipping this task (required tokens > {MAX_NO_TOKENS})",
          flush=True)
    return []
