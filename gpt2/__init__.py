import sys
from transformers import set_seed, GPT2TokenizerFast

# Configuration
# MODEL = 'gpt2'
MODEL = 'gpt2-large'
# MODEL = 'gpt2-xl'
NO_GENERATED_RESULTS = 10
SEED = 42  # None

# Constants
MAX_NO_TOKENS = 1024
TOKENIZER = GPT2TokenizerFast.from_pretrained(MODEL)
DATA_DIR = './data/evaluation/'

# For reproducability
if SEED is not None:
    set_seed(42)

# Get access to the utils from one folder above
sys.path.append('../')
import utils


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
