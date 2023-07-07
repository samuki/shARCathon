from utils import load_json_data, get_logger
from transformers import set_seed, GPT2TokenizerFast

# Configuration
# DEBUG = False
DEBUG = True
SEED = 42  # None

# MODEL = 'gpt2-small'
MODEL = 'gpt2-large'
# MODEL = 'gpt2-xl'
NO_GENERATED_RESULTS = 10

# LIST_REPR = 'tiny'  # separate entries with spaces and lines with ;
LIST_REPR = 'normal'  # separate entries with , and lines with ],[

# Constants
MAX_NO_TOKENS = 1024
TOKENIZER = GPT2TokenizerFast.from_pretrained(MODEL)
TRAIN_DATA_DIR = './data/training/'
DATA_DIR = './data/evaluation/'

# For reproducability
if SEED is not None:
    set_seed(42)


def get_expected_result(data):
    res = ""
    d = data['test'][0]['output']
    in_v = minimize_list_of_list(d)
    res += "Out: " + in_v

    return res


def minimize_list_of_list(ll):
    if LIST_REPR == 'normal':
        res = "["
    else:
        res = ""

    for lst in ll:
        if LIST_REPR == 'tiny':
            # Remove commas to reduce num of tokens required
            # and separate list of lists with ;
            res += " ".join(map(str, lst)) + "; "
        elif LIST_REPR == 'normal':
            res += ",".join(map(str, lst)) + "],["

    res = res[:-2]
    if LIST_REPR == 'normal':
        res = "[" + res + "]"

    return res
