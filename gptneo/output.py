import json

DATA = []


def add_datapoint(prompt, result, exp_result):
    dp = {
        'prompt': prompt,
        'result': result,
        'exp_result': exp_result,
    }
    DATA.append(dp)


def dump_data(path):
    with open(path, 'w+') as f:
        json.dump(DATA, f)
