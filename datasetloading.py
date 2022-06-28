import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sympy import im

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

    def periodic_data_padding(self):
        data_dim = self.data.shape[1]-1
        total_data_dim = int(2**(2**np.ceil(np.log2(np.log2(data_dim)))))
        num_added_columns = 0
        for i in range(data_dim, total_data_dim):
            self.data[str(i)] = self.data[str((num_added_columns % data_dim))]
            num_added_columns += 1
            
    def __getitem__(self, index):
        label = self.data.iloc[index]['label']
        image = self.data.iloc[index][1:]
        return torch.tensor(image), torch.tensor(label)

    def __len__(self):
        return len(self.data)

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

    def load(self):
        self.training_data = read_csv("./training_data.csv")
        self.test_data = read_csv("./test_data.csv")        


if __name__=="__main__":
    TRAINSIZE =10000
    VALSIZE=1000
    TESTSIZE=1000
    SEED=1234
    from torch.utils.data import random_split
    loader = TrainTestLoader('./mnist_train.csv', './mnist_test.csv')
    loader.scale_pca(n_dim=30)
    training_set, test_set = loader.return_dataset()
    training_set.filter([0, 1])
    training_set, validation_set = random_split(training_set, [TRAINSIZE, len(training_set)-TRAINSIZE], generator=torch.Generator().manual_seed(SEED))
    validation_set, _ = random_split(validation_set, [VALSIZE, len(validation_set)-VALSIZE], generator=torch.Generator().manual_seed(SEED))
    test_set, _ = random_split(test_set, [TESTSIZE, len(test_set)-TESTSIZE], generator=torch.Generator().manual_seed(SEED))

    # Create data loaders for our datasets; shuffle for training, not for test
    training_loader = DataLoader(training_set, batch_size=25, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    validation_loader = DataLoader(validation_set, batch_size=25, shuffle=False, generator=torch.Generator().manual_seed(SEED))
    test_loader = DataLoader(test_set, batch_size=25, shuffle=False, generator=torch.Generator().manual_seed(SEED))

    # ====================================

    # Report split sizes
    print('Training set has {} instances'.format(len(training_set)))
    print('validation set has {} instances'.format(len(validation_set)))
    print('test set has {} instances'.format(len(test_set)))