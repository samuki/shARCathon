import config
import json
import os

import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt


# PATH to dataset to visualize
DATA_PATH = config.TRAIN_PATH
# Indices of images to visualize
IDX_LIST = [1, 2]

def plot_examples(data_path, idx_list):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """
    training_tasks = sorted(os.listdir(data_path))
    for i in idx_list:
        task_file = str(data_path / training_tasks[i])
        with open(task_file, 'r') as f:
            task = json.load(f)
        cmap = colors.ListedColormap(
            ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
            '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
        norm = colors.Normalize(vmin=0, vmax=9)
        fig, axs = plt.subplots(1, 4, figsize=(15,15))
        axs[0].imshow(task['train'][0]['input'], cmap=cmap, norm=norm)
        axs[0].axis('off')
        axs[0].set_title('Train Input')
        axs[1].imshow(task['train'][0]['output'], cmap=cmap, norm=norm)
        axs[1].axis('off')
        axs[1].set_title('Train Output')
        axs[2].imshow(task['test'][0]['input'], cmap=cmap, norm=norm)
        axs[2].axis('off')
        axs[2].set_title('Test Input')
        axs[3].imshow(task['test'][0]['output'], cmap=cmap, norm=norm)
        axs[3].axis('off')
        axs[3].set_title('Test Output')
        plt.tight_layout()
        plt.show()
            
def main():
    plot_examples(DATA_PATH, IDX_LIST)

if __name__ == '__main__':
    main()