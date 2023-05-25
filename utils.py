import numpy as np
import cv2
from keras.utils import to_categorical
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt

# Own imports
import config
import json
import os


def flt(x): return np.float32(x)


def npy(x): return x.cpu().detach().numpy()


def itg(x): return np.int32(np.round(x))


def transform_dim(inp_dim, outp_dim, test_dim):
    return (test_dim[0]*outp_dim[0]/inp_dim[0],
            test_dim[1]*outp_dim[1]/inp_dim[1])

def resize(x, test_dim, inp_dim):
    if inp_dim == test_dim:
        return x
    else:
        return cv2.resize(flt(x), inp_dim,
                          interpolation=cv2.INTER_AREA)

def replace_values(a, d):
    return np.array([d.get(i, -1) for i in range(a.min(), a.max() + 1)])[a - a.min()]


def repeat_matrix(a):
    return np.concatenate([a]*((config.SIZE // len(a)) + 1))[:config.SIZE]


def get_new_matrix(X):
    if len(set([np.array(x).shape for x in X])) > 1:
        X = np.array([X[0]])
    return X


def get_outp(outp, dictionary=None, replace=True):
    if replace:
        outp = replace_values(outp, dictionary)
    outp_matrix_dims = outp.shape
    outp_probs_len = outp.shape[0]*outp.shape[1]*10
    outp = to_categorical(outp.flatten(),
                          num_classes=10).flatten()
    return outp, outp_probs_len, outp_matrix_dims


def load_data(path):
    files = sorted(os.listdir(path))
    tasks = []
    for task_file in files:
        with open(str(path / task_file), 'r') as f:
            task = json.load(f)
        tasks.append(task)  
    Xs_train, Xs_test, ys_train, ys_test = [], [], [], []
    for task in tasks:
        X_train, X_test, y_train, y_test = [], [], [], []
        for pair in task["train"]:
            X_train.append(pair["input"])
            y_train.append(pair["output"])  
        for pair in task["test"]:
            X_test.append(pair["input"])
            y_test.append(pair["input"])
        Xs_train.append(X_train)
        Xs_test.append(X_test)
        ys_train.append(y_train)
        ys_test.append(y_test)
        
    return Xs_train, Xs_test, ys_train, ys_test

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
        
def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred
