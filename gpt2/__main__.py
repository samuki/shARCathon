
from .basic import main as basic_main
from .finetuned import main as finetuned_main
from .prompts import PROMPT_KINDS

import sys

WRONG_PROMPTKIND_MSG = "Unexpected argument! Please use one of:\n" + \
    "\n".join(["python -m gpt2 basic " + kind for kind in PROMPT_KINDS])
WRONG_ARGS_MSG = """Unexpected argument! Please use one of:
python -m gpt2 finetuned
python -m gpt2 basic"""
MISSING_ARGS_MSG = """Missing argument! Please use one of:
python -m gpt2 finetuned
python -m gpt2 basic"""

if __name__ == '__main__':
    print(sys.argv)
    if (len(sys.argv) <= 1):
        print(MISSING_ARGS_MSG)
        exit(1)
    if (sys.argv[1] == "basic"):
        kind = sys.argv[2] if len(sys.argv) > 2 else 'basic'
        if kind not in PROMPT_KINDS:
            print(WRONG_PROMPTKIND_MSG)
            exit(1)
        basic_main(kind=kind)
    elif (sys.argv[1] == "finetuned"):
        finetuned_main()
    else:
        print(WRONG_ARGS_MSG)
        exit(1)
