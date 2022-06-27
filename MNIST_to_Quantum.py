import numpy as np
import os
import torch as T
import torchvision.datasets as tds
import torchvision.transforms as transforms


class Get_MNIST_Quantum_States:
    def __init__(self, mnist_size):
        self.i = T.complex(T.tensor(0, dtype=T.float),
                           T.tensor(1, dtype=T.float))
        self.mnist_size = mnist_size
        mnist_path = './mnist datas'
        if not os.path.exists(mnist_path):
            os.makedirs(mnist_path)
        trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,)), transforms.Resize(mnist_size)])

        self.train_set = tds.MNIST(root=mnist_path, train=True,
                                   transform=trans, download=True)
        self.test_set = tds.MNIST(root=mnist_path, train=False,
                                  transform=trans, download=True)

    def Get_MNIST(self, N_train, N_test, labels):
        train_indices = np.array([idx for idx, target in enumerate(
            self.train_set.targets) if target in labels])
        test_indices = np.array([idx for idx, target in enumerate(
            self.test_set.targets) if target in labels])
        train_indices = np.random.choice(
            train_indices, N_train, replace=False)
        test_indices = np.random.choice(
            test_indices, N_test, replace=False)
        self.train_sample = T.utils.data.Subset(self.train_set, train_indices)
        self.test_sample = T.utils.data.Subset(self.test_set, test_indices)

    def Real_Complex_Encoding(self):
        quantum_state_size = int(0.5*self.mnist_size**2)
        self.train_quantum_states = T.zeros(
            len(self.train_sample), quantum_state_size, quantum_state_size, dtype=T.cfloat)
        self.test_quantum_states = T.zeros(
            len(self.test_sample), quantum_state_size, quantum_state_size, dtype=T.cfloat)
        for (train_idx, data) in enumerate(self.train_sample):
            x = T.reshape(T.squeeze(data[0]), (-1,))
            qs = T.zeros(quantum_state_size, 1, dtype=T.cfloat)
            for i in range(quantum_state_size):
                qs[i] = x[i]+self.i*x[quantum_state_size+i]
            qs = qs/T.sqrt(T.sum(T.abs(qs)**2))
            qs = T.matmul(qs, T.transpose(T.conj(qs), 0, 1))
            self.train_quantum_states[train_idx] = qs
        for (test_idx, data) in enumerate(self.test_sample):
            x = T.reshape(T.squeeze(data[0]), (-1,))
            qs = T.zeros(quantum_state_size, 1, dtype=T.cfloat)
            for i in range(quantum_state_size):
                qs[i] = x[i]+self.i*x[quantum_state_size+i]
            qs = qs/T.sqrt(T.sum(T.abs(qs)**2))
            qs = T.matmul(qs, T.transpose(T.conj(qs), 0, 1))
            self.test_quantum_states[test_idx] = qs

    def Real_Encoding(self):
        quantum_state_size = self.mnist_size**2
        self.train_quantum_states = T.zeros(
            len(self.train_sample), quantum_state_size, quantum_state_size, dtype=T.cfloat)
        self.test_quantum_states = T.zeros(
            len(self.test_sample), quantum_state_size, quantum_state_size, dtype=T.cfloat)
        for (train_idx, data) in enumerate(self.train_sample):
            x = T.reshape(T.squeeze(data[0]), (-1,))
            qs = T.zeros(quantum_state_size, 1, dtype=T.cfloat)
            for i in range(quantum_state_size):
                qs[i] = x[i]
            qs = qs/T.sqrt(T.sum(T.abs(qs)**2))
            qs = T.matmul(qs, T.transpose(T.conj(qs), 0, 1))
            self.train_quantum_states[train_idx] = qs
        for (test_idx, data) in enumerate(self.test_sample):
            x = T.reshape(T.squeeze(data[0]), (-1,))
            qs = T.zeros(quantum_state_size, 1, dtype=T.cfloat)
            for i in range(quantum_state_size):
                qs[i] = x[i]
            qs = qs/T.sqrt(T.sum(T.abs(qs)**2))
            qs = T.matmul(qs, T.transpose(T.conj(qs), 0, 1))
            self.test_quantum_states[test_idx] = qs
