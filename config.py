from pathlib import Path

# Paths config
TRAIN_PATH = Path('data/toy')
VALID_PATH = Path('data/evaluation')
OUT_PRED_PATH = Path('results/predictions.csv')

# CNN config
SIZE = 1000
EPOCHS = 2
CONV_OUT_1 = 50
CONV_OUT_2 = 100
CONV_IN = 3
KERNEL_SIZE = 3
BATCH_SIZE = 128
