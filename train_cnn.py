import time
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import pandas as pd

# Own imports
import dataset
from models import cnn
import config
import utils

def main():
    idx = 0
    start = time.time()
    test_predictions = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Xs_train, Xs_test, ys_train, ys_test, labels = utils.load_data(config.TRAIN_PATH)
    print("Data shapes")
    print(len(Xs_train))
    print(len(Xs_test))
    print(len(ys_train))
    print(len(ys_test))
    print(len(labels))
    for X_train, y_train in zip(Xs_train, ys_train):
        print("TASK " + str(idx + 1))
        train_set = dataset.ARCDataset(X_train, y_train, stage="train")
        train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
        
        inp_dim = np.array(X_train[0]).shape
        outp_dim = np.array(y_train[0]).shape
        network = cnn.BasicCNNModel(inp_dim, outp_dim).to(device)
        optimizer = Adam(network.parameters(), lr=0.01)
        
        for _ in range(config.EPOCHS):
            for train_batch in train_loader:
                train_X, train_y, out_d, _, _ = train_batch
                train_preds = network.forward(train_X.to(device), out_d.to(device))
                print(train_preds.shape)
                train_loss = nn.MSELoss()(train_preds, train_y.to(device))
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

        end = time.time()        
        print("Train loss: " + str(np.round(train_loss.item(), 3)) + "   " +\
            "Total time: " + str(np.round(end - start, 1)) + " s" + "\n")
        
        X_test = np.array([utils.resize(utils.flt(X), np.shape(X), inp_dim) for X in Xs_test[idx-1]])
        for X in X_test:
            test_dim = np.array(torch.Tensor(X)).shape
            test_preds = utils.npy(network.forward(torch.Tensor(X).unsqueeze(0).to(device), out_d.to(device)))
            test_preds = np.argmax(test_preds.reshape((10, *outp_dim)), axis=0)
            test_predictions.append(utils.itg(utils.resize(test_preds, np.shape(test_preds),
                                            tuple(utils.itg(utils.transform_dim(inp_dim,
                                                                    outp_dim,
                                                                    test_dim))))))
        idx += 1
        
    test_predictions = [[list(pred) for pred in test_pred] for test_pred in test_predictions]
    test_dict = dict(zip(labels, test_predictions))
    utils.save_predictions(test_dict)
    

        
if __name__ == '__main__':
    main()