from datetime import datetime
import logging
import numpy as np
import cv2
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

import os
import json

# Own imports
import config


#----------------------------- Logging -------------------------------------- #

def get_logger():
    now = datetime.now()  # current date and time
    log_name = now.strftime("%Y%m%d%H%M%S")
    log_name = f"{log_name}.log"
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_path = os.path.join(log_dir, log_name)
    logging.basicConfig(level=logging.INFO, filename=log_file_path, filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger()

# --------------------------- Helpers for the CNN ---------------------------------- #

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


def load_results_json(path):
    files = sorted(os.listdir(path))
    tasks = {}
    for task in files:
        with open(str(path / task), 'r', encoding='utf-8') as f:
            task = json.load(f)
        tasks.update(task)
    return tasks


def load_train_json(path):
    files = sorted(os.listdir(path))
    tasks = {}
    for task in files:
        taskname = task.split('.')[0]
        with open(str(path / task), 'r', encoding='utf-8') as f:
            task = json.load(f)
            test_out = {taskname: task["test"][0]["output"]}
        tasks.update(test_out)
    return tasks


def load_data(path):
    files = sorted(os.listdir(path))
    tasks = []
    labels = []
    for task_file in files:
        taskname = task_file.split('.')[0]
        labels.append(taskname)
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
        
    return Xs_train, Xs_test, ys_train, ys_test, labels

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


def convert(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError


def save_predictions(preds):
    for pred in preds:
        pred_dict = {pred: [[int(i) for i in sublist] for sublist in preds[pred]]}
        with open(config.OUT_PRED_PATH + pred+'_out'+'.json', 'w') as fp:
            json.dump(pred_dict, fp)
            
  
 # --------------------------- Helpers for Language models ---------------------------------- #    
        
def load_json_data(folder):
    json_files = [pos_json for pos_json in os.listdir(folder) if pos_json.endswith('.json')]
    data = {}
    for js in json_files:
        with open(os.path.join(folder, js)) as json_file:
            data[js] = json.load(json_file)
    return data
            
 # --------------------------- Helpers for Plotting ---------------------------------- #    
    
def plot_2d_grid(task_name,mode_name, data):
    cvals  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    colors = ["black", "dodgerblue", "red", "lightgreen", "yellow", "grey", "magenta", "orange", "lightblue", "brown"]
    norm=plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

    fig, axs = plt.subplots(1, 3, figsize=(5, len(data['test']) * 3))
    print(axs.shape)
    axs[0].set_title('Test Input')
    axs[0].set_xticks([]); axs[0].set_yticks([])
    axs[0].imshow(np.array(data['test'][0]['input']), cmap=cmap, vmin=0, vmax=9)
    axs[1].set_title('Test Output')
    axs[1].set_xticks([]); axs[1].set_yticks([])
    axs[1].imshow(np.array(data['test'][0]['output']), cmap=cmap, vmin=0, vmax=9)
    # plot gpt output if present
    if data['gpt_output'] is not None:
        axs[2].set_title(mode_name+' Output')
        axs[2].set_xticks([]); axs[2].set_yticks([])
        axs[2].imshow(np.array(data['gpt_output']), cmap=cmap, vmin=0, vmax=9) 
    else:
        axs[2].axis('off')
    plt.tight_layout()
    plt.savefig('examples/'+task_name+'_'+mode_name+'_output'+'.png')

    fig, axs = plt.subplots(len(data['train']), 2, figsize=(5, len(data['train']) * 3))
    for i, example in enumerate(data['train']):
        axs[i, 0].set_title(f'Training Input {i}', fontsize=20)
        axs[i, 0].set_xticks([]); axs[i, 0].set_yticks([])
        axs[i, 0].imshow(np.array(example['input']), cmap=cmap, vmin=0, vmax=9)
        axs[i, 1].set_title(f'Training Output {i}', fontsize=20)
        axs[i, 1].set_xticks([]); axs[i, 1].set_yticks([])
        axs[i, 1].imshow(np.array(example['output']), cmap=cmap, vmin=0, vmax=9)
    plt.tight_layout()
    plt.savefig('examples/'+task_name+'_task'+'.png')
    #plt.show()

def plot_single_output(task_name,mode_name, data):
    cvals  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    colors = ["black", "dodgerblue", "red", "lightgreen", "yellow", "grey", "magenta", "orange", "lightblue", "brown"]
    norm=plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
    fig, axs = plt.subplots(1, 1, figsize=(len(data['test'])*3, len(data['test'])*3))
    axs.set_title(mode_name+' Output', fontsize=20)
    axs.set_xticks([]); axs.set_yticks([])
    axs.imshow(np.array(data['gpt_output']), cmap=cmap, vmin=0, vmax=9) 
    plt.tight_layout()
    plt.savefig('examples/'+task_name+'_'+mode_name+'_only_output'+'.png')