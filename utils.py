import numpy as np
from keras.utils import to_categorical
import cv2

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
