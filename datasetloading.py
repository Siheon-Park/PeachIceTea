import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torch
from pandas import concat
from pandas import read_csv

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class DataframeDataset(Dataset):
    def __init__(self, data):
        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            self.data = read_csv(data)

    def filter(self, labels):
        data_list  = []
        for label in labels:
            data_list.append(self.data[self.data["label"]==label])
        self.data = concat(data_list)

    def __getitem__(self, index):
        label = self.data.iloc[index]['label']
        image = self.data.iloc[index][1:]
        return torch.tensor(image), torch.tensor(label)

    def __len__(self):
        return len(self.data)

def preprocess_train_and_test_dataset(path_to_train, path_to_test, pca_dim:int=5, save:bool=False, dummy:bool=False):
    training_data = read_csv(path_to_train)
    test_data = read_csv(path_to_test)

    if not dummy:
        training_label = training_data.iloc[:,0]
        training_data = training_data.iloc[:,1:]
        test_label = test_data.iloc[:,0]
        test_data = test_data.iloc[:,1:]

        scaler = StandardScaler()
        training_data = scaler.fit_transform(training_data)
        test_data = scaler.transform(test_data)

        pca = PCA(n_components=pca_dim)
        training_data = pca.fit_transform(training_data)
        test_data = pca.transform(test_data)

        training_data = pd.DataFrame(training_data)
        training_data.insert(0, 'label', training_label)
        if save:
            training_data.to_csv("training_data.csv", index = False)

        training_data = pd.DataFrame(test_data)
        training_data.insert(0, 'label', test_label)
        if save:
            training_data.to_csv("test_data.csv", index = False)

    return DataframeDataset(training_data), DataframeDataset(training_data)

class TrainTestLoader(object):
    def __init__(self, path_to_train, path_to_test):
            self.training_data = read_csv(path_to_train)
            self.test_data = read_csv(path_to_test)

    def scale_pca(self, n_dim):
        training_label = self.training_data.iloc[:,0]
        training_data = self.training_data.iloc[:,1:]
        test_label = self.test_data.iloc[:,0]
        test_data = self.test_data.iloc[:,1:]

        scaler = StandardScaler()
        training_data = scaler.fit_transform(training_data)
        test_data = scaler.transform(test_data)

        pca = PCA(n_components=n_dim)
        
        training_data = pca.fit_transform(training_data)
        test_data = pca.transform(test_data)

        training_data = pd.DataFrame(training_data)
        training_data.insert(0, 'label', training_label)

        test_data = pd.DataFrame(test_data)
        test_data.insert(0, 'label', test_label)

        self.training_data = training_data
        self.test_data = test_data

    def return_dataset(self):
        return DataframeDataset(self.training_data), DataframeDataset(self.training_data)

    def save(self):
        self.training_data.to_csv("./training_data.csv", index = False)
        self.test_data.to_csv("./test_data.csv", index = False)
