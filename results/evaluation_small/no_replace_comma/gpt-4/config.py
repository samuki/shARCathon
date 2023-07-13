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

#OPENAI_ENDPOINT = "Completion"
OPENAI_ENDPOINT = "Chat"

# Select experiments to run
PATH_SELECTION = VALID_SMALL_PATH

# GPT Models

#GPT_MODEL = "gpt-3.5-turbo-16k"
#GPT_MODEL = "code-davinci-002"
#GPT_MODEL = "text-davinci-002"
#GPT_MODEL = "gpt-3.5-turbo"
GPT_MODEL = "gpt-4"

# Max token configuration to avoid length limit
MAX_TOKENS = 4000
#MAX_TOKENS = 6000
#MODEL_MAX_TOKENS = 16000
MODEL_MAX_TOKENS = 8192

# OpenAI API parameters
TEMPERATURE = 1
TOP_P = 1
LOG_PROBS = 5
# Replace , since the model can solve this too.
SPARSE_MATRIX = False
COMPRESS = False

REPLACE_COMMA = False
REPLACE_SPACE = False
REPLACE_SPACE2 = False
SEMICOLON = False
BRACKETS = False

REPLACE_NUMBER_COLOR = False
REPLACE_NUMBER_WORD = False
REPLACE_NUMBER_BINARY = False
REPLACE_NUMBER_LEET = False
REPLACE_NUMBER_CHAR = True
REPLACE_NUMBER_SP_CHAR = False
# Prompt template
#PROMPT_TEMPLATE = prompt_toolkit.BETTER_STRUCTURE_PREMEABLE
PROMPT_TEMPLATE = prompt_toolkit.NO_PREMEABLE
#PROMPT_TEMPLATE = prompt_toolkit.PREMEABLE

CREATE_DESCRIPTION = False
USE_DESCRIPTION = False
DESCRIPTION_PATH = Path('results/evaluation_small_description/replace_comma/gpt-3.5-turbo-16k/')
#DESCRIPTION_PATH = Path('results/evaluation_small_description/replace_comma/gpt-4/')

SELF_CONSISTENCY = False
DOUBLE_SELF_CONSISTENCY = False

ADD_TRAINING = False
#SELECTED_TRAINING_TASKS = ['0d3d703e.json', '1e0a9b12.json']
SELECTED_TRAINING_TASKS = ['0d3d703e.json', '1e0a9b12.json', '3c9b0459.json']