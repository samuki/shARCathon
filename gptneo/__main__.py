
from .basic import main as basic_main
from .finetuned import main as finetuned_main
from .prompts import PROMPT_KINDS
from . import LIST_REPR_KINDS, DEBUG, get_logger

import sys

WRONG_LISTREPR_MSG = "Unexpected argument! Please use one of:\n" + \
    "\n".join(["python -m gptneo basic basic " + kind for kind in LIST_REPR_KINDS])

WRONG_PROMPTKIND_MSG = "Unexpected argument! Please use one of:\n" + \
    "\n".join(["python -m gptneo basic " + kind for kind in PROMPT_KINDS])

WRONG_ARGS_MSG = """Unexpected argument! Please use one of:
python -m gptneo finetuned
python -m gptneo basic"""

MISSING_ARGS_MSG = """Missing argument! Please use one of:
python -m gptneo finetuned
python -m gptneo basic"""

if __name__ == '__main__':
    logger = get_logger()
    if DEBUG:
        logger.info = print

    logger.info(f"\t|> Called with following args: {' '.join(sys.argv)}")

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

    if (sys.argv[1] == "basic"):
        basic_main(logger, kind=kind, list_kind=line_kind)
    elif (sys.argv[1] == "finetuned"):
        finetuned_main(logger, kind=kind, list_kind=line_kind)
    else:
        print(WRONG_ARGS_MSG)
        exit(1)
