from pathlib import Path

# Paths config
TRAIN_PATH = Path('data/training')
VALID_PATH = Path('data/evaluation')
OUT_PRED_PATH = Path('results/')

# CNN config
SIZE = 1000
EPOCHS = 2
CONV_OUT_1 = 50
CONV_OUT_2 = 100
CONV_IN = 3
KERNEL_SIZE = 3
BATCH_SIZE = 128


# GPT config
GPT_MODEL = "gpt-3.5-turbo-16k"
#GPT_MODEL = "gpt-3.5-turbo"
#GPT_MODEL = "gpt-4"
TEMPERATURE = 0.8
MAX_TOKENS = 4500