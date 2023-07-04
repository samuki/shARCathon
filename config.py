from pathlib import Path

import llm.prompt_toolkit as prompt_toolkit

# ---------------------  PATHS CONFIG ------------------

TRAIN_PATH = Path('data/training')
TRAIN_SMALL_PATH = Path('data/training_small')
TRAIN_MEDIUM_PATH = Path('data/training_medium')
TRAIN_LARGE_PATH = Path('data/training_large')

VALID_PATH = Path('data/evaluation')
VALID_SMALL_PATH = Path('data/evaluation_small')
VALID_MEDIUM_PATH = Path('data/evaluation_medium')
VALID_LARGE_PATH = Path('data/evaluation_large')

OUT_PRED_PATH = Path('results/')

# --------------------- CNN CONFIG ---------------------

SIZE = 1000
EPOCHS = 2
CONV_OUT_1 = 50
CONV_OUT_2 = 100
CONV_IN = 3
KERNEL_SIZE = 3
BATCH_SIZE = 128

# ---------------------- GPT CONFIG ---------------------

# Select experiments to run
PATH_SELECTION = VALID_SMALL_PATH

# GPT Models
GPT_MODEL = "gpt-3.5-turbo-16k"
#GPT_MODEL = "gpt-3.5-turbo"
#GPT_MODEL = "gpt-4"

# Max token configuration to avoid length limit
MAX_TOKENS = 6000
MODEL_MAX_TOKENS = 16000

# OpenAI API parameters
TEMPERATURE = 1
TOP_P = 1

# Replace , since the model can solve this too.
REPLACE_COMMA = False

# Prompt template
PROMPT_TEMPLATE = prompt_toolkit.BETTER_STRUCTURE_PREMEABLE
#PROMPT_TEMPLATE = prompt_toolkit.NO_PREMEABLE
