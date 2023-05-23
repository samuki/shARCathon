from pathlib import Path

# Paths config
TRAIN_PATH = Path('data/training')
VALID_PATH = Path('data/evaluation')
RESULT_PATH = Path('results/result.csv')

# CNN config
SIZE = 1000
EPOCHS = 50
CONV_OUT_1 = 50
CONV_OUT_2 = 100
CONV_IN = 3
KERNEL_SIZE = 3
BATCH_SIZE = 128
