import config
import utils

# PATH to dataset to visualize
DATA_PATH = config.TRAIN_PATH
# Indices of images to visualize
IDX_LIST = [1]
         
def main():
    utils.plot_examples(DATA_PATH, IDX_LIST)

if __name__ == '__main__':
    main()