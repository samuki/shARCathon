from utils import load_json_data, get_logger
from transformers import set_seed, GPT2TokenizerFast
import numpy as np

# Configuration
DEBUG = False
# DEBUG = True
SEED = 42  # None

# MODEL = 'gpt2-small'
MODEL = 'gpt2-large'
# MODEL = 'gpt2-xl'
NO_GENERATED_RESULTS = 5

LIST_REPR_KINDS = [
    'tiny',  # separate entries with spaces and lines with ;
    'small',  # separate entries with spaces and lines with ] [
    'normal',  # separate entries with , and lines with ],[
]

# Constants
MAX_NO_TOKENS = 1024
TOKENIZER = GPT2TokenizerFast.from_pretrained(MODEL)
TRAIN_DATA_DIR = './data/training/'
DATA_DIR = './data/evaluation/'

# For reproducability
if SEED is not None:
    set_seed(42)


def get_expected_result(data, list_kind='small'):
    res = ""
    d = data['test'][0]['output']
    in_v = minimize_list_of_list(d, list_kind=list_kind)
    res += "Out: " + in_v

    return res


def select_best_answer(answers, list_kind='small'):
    if len(answers) <= 0:
        return None
    if len(answers) == 1:
        return answers[0]

    if list_kind == 'tiny':
        exp_chars = [str(i) for i in range(0, 10)] + [';', ' ']
    elif list_kind == 'small':
        exp_chars = [str(i) for i in range(0, 10)] + [' ', '[', ']']
    elif list_kind == 'normal':
        exp_chars = [str(i) for i in range(0, 10)] + [',', '[', ']']
    else:
        exp_chars = [str(i) for i in range(0, 10)] + [',', '[', ']', ';']

    # Simple approach: Pick the answer with the least 'non-expected chars'
    scores = len(answers) * [0]
    for i, answer in enumerate(answers):
        score = 0
        for c in answer:
            if c not in exp_chars:
                score += 1
        scores[i] = score
    return answers[np.argmin(scores)]


def minimize_list_of_list(ll, list_kind='small'):
    if list_kind == 'normal' or list_kind == 'small':
        res = "["
    else:
        res = ""

    for lst in ll:
        if list_kind == 'tiny':
            # Remove commas to reduce num of tokens required
            # and separate list of lists with ;
            res += " ".join(map(str, lst)) + "; "
        elif list_kind == 'normal':
            res += ",".join(map(str, lst)) + "],["
        elif list_kind == 'small':
            res += " ".join(map(str, lst)) + "] ["

    res = res[:-2]
    if list_kind == 'normal' or list_kind == 'small':
        res = "[" + res + "]"

    return res
