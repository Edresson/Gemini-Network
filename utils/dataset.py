import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import stack
import numpy as np
import pandas as pd
import random
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class Dataset(Dataset):
    """
    Class for load a train and test from dataset generate by import_librispeech.py and others
    """
    def __init__(self, c,  train=True, test=False):
        # set random seed
        random.seed(c['seed'])
        torch.manual_seed(c['seed'])
        torch.cuda.manual_seed(c['seed'])
        np.random.seed(c['seed'])
        self.c = c
        self.train = train
        self.dataset_csv = c.dataset['train_csv'] if train else c.dataset['eval_csv']
        if test:
            self.dataset_csv = c.dataset['test_csv']
        assert os.path.isfile(self.dataset_csv),"Test or Train CSV file don't exists! Fix it in config.json"
        
        # read csvs
        self.dataset_list = pd.read_csv(self.dataset_csv, sep=',').values
        # invert emb1 and emb2 
        for data in self.dataset_list:
            self.dataset_list.append(data[:4]+[data[6], data[5]])

    def __getitem__(self, idx):
        self.dataset_list[idx][0]
        class_name = int(self.dataset_list[idx][1])
        # map class 1 to 0 and class 2 to 1 for binnary classification
        if class_name == 1:
            class_name = 0
        else:
            class_name = 1

        feature = torch.FloatTensor(self.dataset_list[idx][5]+self.dataset_list[idx][6])
        target = torch.FloatTensor([class_name])
        return feature, target

    def __len__(self):
        return len(self.dataset_list)

def train_dataloader(c):
    return DataLoader(dataset=Dataset(c, train=True),
                          batch_size=c.train_config['batch_size'],
                          shuffle=True,
                          num_workers=c.train_config['num_workers'],
                          collate_fn=own_collate_fn,
                          pin_memory=True,
                          drop_last=True,
                          sampler=None)

def eval_dataloader(c):
    return DataLoader(dataset=Dataset(c, train=False),
                          collate_fn=own_collate_fn, batch_size=c.test_config['batch_size'], 
                          shuffle=False, num_workers=c.test_config['num_workers'])


def test_dataloader(c):
    return DataLoader(dataset=Dataset(c, train=False, test=True),
                          collate_fn=own_collate_fn, batch_size=c.test_config['batch_size'], 
                          shuffle=False, num_workers=c.test_config['num_workers'])

def own_collate_fn(batch):
    features = []
    targets = []
    for feature, target in batch:
        features.append(feature)
        targets.append(target)
    # list to tensor
    targets = stack(targets, dim=0)
    features = stack(features, dim=0)
    return features, targets