# Own imports
import utils
import config

from torch.utils.data import Dataset 
import numpy as np


class ARCDataset(Dataset):
    def __init__(self, X, y, stage="train"):
        self.X = utils.get_new_matrix(X)
        self.X = utils.repeat_matrix(self.X)      
        self.stage = stage
        if self.stage == "train":
            self.y = utils.get_new_matrix(y)
            self.y = utils.repeat_matrix(self.y)
        
    def __len__(self):
        return config.SIZE
    
    def __getitem__(self, idx):
        inp = self.X[idx]
        if self.stage == "train":
            outp = self.y[idx]

        if idx != 0:
            rep = np.arange(10)
            orig = np.arange(10)
            np.random.shuffle(rep)
            dictionary = dict(zip(orig, rep))
            inp = utils.replace_values(inp, dictionary)
            if self.stage == "train":
                outp, outp_probs_len, outp_matrix_dims = utils.get_outp(outp, dictionary)
                
        if idx == 0:
            if self.stage == "train":
                outp, outp_probs_len, outp_matrix_dims = utils.get_outp(outp, None, False)
        
        return inp, outp, outp_probs_len, outp_matrix_dims, self.y