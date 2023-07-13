
from .basic import main as basic_main
from .finetuned import main as finetuned_main
from .prompts import PROMPT_KINDS
from . import LIST_REPR_KINDS

import sys
import os
import time

WRONG_LISTREPR_MSG = "Unexpected argument! Please use one of:\n" + \
    "\n".join(["python -m gpt2 basic basic " + kind for kind in LIST_REPR_KINDS])

WRONG_PROMPTKIND_MSG = "Unexpected argument! Please use one of:\n" + \
    "\n".join(["python -m gpt2 basic " + kind for kind in PROMPT_KINDS])

WRONG_ARGS_MSG = """Unexpected argument! Please use one of:
python -m gpt2 finetuned
python -m gpt2 basic"""

MISSING_ARGS_MSG = """Missing argument! Please use one of:
python -m gpt2 finetuned
python -m gpt2 basic"""

if __name__ == '__main__':
    print(f"\t|> Called with following args: {' '.join(sys.argv)}")

    if (len(sys.argv) <= 1):
        print(MISSING_ARGS_MSG)
        exit(1)

    kind = sys.argv[2] if len(sys.argv) > 2 else 'basic'
    if kind not in PROMPT_KINDS:
        print(WRONG_PROMPTKIND_MSG)
        exit(1)

    line_kind = sys.argv[3] if len(sys.argv) > 3 else 'small'
    if line_kind not in LIST_REPR_KINDS:
        print(WRONG_LISTREPR_MSG)
        exit(1)

    json_path = f'./results/{"".join(sys.argv).replace("/", "")}/{time.time()}.json'
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    if (sys.argv[1] == "basic"):
        basic_main(json_path, kind=kind, list_kind=line_kind)
    elif (sys.argv[1] == "finetuned"):
        finetuned_main(json_path, kind=kind, list_kind=line_kind)
    else:
        print(WRONG_ARGS_MSG)
        exit(1)
