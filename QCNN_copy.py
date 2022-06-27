import torch as T
import numpy as np
import MNIST_to_Quantum as MQ
import pennylane as qml


class QCNN:
    def __init__(self, n_qubits, n_epoch, batch_size, n_data, stride):
        self.n_qubits = n_qubits
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.n_data = n_data
        self.stride = stride
        # total required layer for QCNN
        self.total_layer = 1
        # information about used qubits in each layer
        self.QCNN_tree = []
        # ancilla qubit location
        self.ancilla_info = []
        # do calculation
        temp_q = self.n_qubits
        temp_qs = list(range(self.n_qubits))
        temp_ancilla = self.n_qubits
        while (temp_q != 1):
            # does not require ancilla
            if temp_q % 2 == 0:
                self.QCNN_tree.append(temp_qs.copy())
                self.ancilla_info.append(-1)
                temp_qs = [int(qs)
                           for (idx, qs) in enumerate(temp_qs) if idx % 2 == 0]
                temp_q = int(temp_q/2)
            # require ancilla
            else:
                self.ancilla_info.append(temp_ancilla)
                temp_qs.append(temp_ancilla)
                temp_ancilla += 1
                self.QCNN_tree.append(temp_qs.copy())
                temp_qs = [int(qs)
                           for (idx, qs) in enumerate(temp_qs) if idx % 2 == 0]
                temp_q = int((temp_q+1)/2)
            self.total_layer += 1

        self.QCNN_tree.append([0])
        self.ancilla_info.append(-1)
        # total qubit required
        self.total_qubit = temp_ancilla
        self.dev = qml.device("default.qubit", wires=self.total_qubit)
        # loss function type
        self.loss = T.nn.MSELoss()
        # generate MNIST dataset
        '''Generate MNIST quantum data set'''
        self.MQ = MQ.Get_MNIST_Quantum_States(16)
        # train set : 1000, test set : 100
        self.MQ.Get_MNIST(1000, 100, [0, 1])  # label 0,1
        self.MQ.Real_Complex_Encoding()
        self.train_qs = self.MQ.train_quantum_states
        self.train_labels = T.tensor(self.MQ.train_labels, dtype=T.float)
        self.test_qs = self.MQ.test_quantum_states
        self.test_labels = T.tensor(self.MQ.test_labels, dtype=T.float)
    def Calculate_Param_Num(self, ansatz_param, pooling_param):
        total_param_num = 0
        for layer in range(self.total_layer):
            qubit_info = np.array(self.QCNN_tree[layer])
            L = len(qubit_info)
            qubit_info_index = np.array(range(L))
            if self.stride == 1:
                qubit_info_splited = [[idx for idx in qubit_info_index if idx % 2 == 0], [
                    idx for idx in qubit_info_index if idx % 2 == 1]]
            else:
                qubit_info_splited = qubit_info_index
            # apply unitary
            for index in qubit_info_splited:
                for idx in index:
                    if qubit_info[idx] != qubit_info[(idx+self.stride) % L]:
                        total_param_num += ansatz_param
            # apply pooling
            for idx in qubit_info_index:
                if idx % 2 == 0:
                    if qubit_info[idx] != qubit_info[(idx+1) % L]:
                        total_param_num += pooling_param
        return total_param_num
    def Do_QCNN(self, ansatz_function, ansatz_param, pooling_function, pooling_param):
        total_param_num = self.Calculate_Param_Num(ansatz_param, pooling_param)
        @qml.qnode(self.dev, interface='torch')
        def qc(thetas, ansatz_f, ansatz_param, pol_f, pol_param, data):
            # insert initial state as data in n_qubits, not total_qubits
            qml.QubitStateVector(data, wires=range(self.n_qubits))
            theta_idx = 0
            for layer in range(self.total_layer):
                qubit_info = np.array(self.QCNN_tree[layer])
                L = len(qubit_info)
                qubit_info_index = np.array(range(L))
                if self.stride == 1:
                    qubit_info_splited = [[idx for idx in qubit_info_index if idx % 2 == 0], [
                        idx for idx in qubit_info_index if idx % 2 == 1]]
                else:
                    qubit_info_splited = qubit_info_index
                # apply unitary
                for index in qubit_info_splited:
                    for idx in index:
                        if qubit_info[idx] != qubit_info[(idx+self.stride) % L]:
                            ansatz_f(
                                thetas[theta_idx:theta_idx+ansatz_param], wires=[qubit_info[idx], qubit_info[(idx+self.stride) % L]])
                            theta_idx += ansatz_param
                # apply pooling
                for idx in qubit_info_index:
                    if idx % 2 == 0:
                        if qubit_info[idx] != qubit_info[(idx+1) % L]:
                            pol_f(thetas[theta_idx:theta_idx+pol_param],
                                  wires=[qubit_info[idx], qubit_info[(idx+1) % L]])
                            theta_idx += pol_param
            return qml.expval(qml.PauliZ(0))
        thetas = T.tensor(2*np.pi*np.random.rand(total_param_num),
                          dtype=T.float, requires_grad=True)
        # --------------------optimizer : learning rate and optimizer type should be tested!!
        opt = T.optim.Adam([thetas], lr=0.1)
        loss_history = []
        for ep in range(self.n_epoch):
            opt.zero_grad()
            batch = np.random.choice(
                np.arange(self.n_data), self.batch_size, replace=False)
            # ---------------------------you have to define train quantum states!!
            batch_data = self.train_qs[batch]
            # ---------------------------- you have to define train labels!!
            # --------------------------- labels must be +1 or -1
            batch_labels = self.train_labels[batch]

            batch_results = T.zeros(self.batch_size, dtype=T.float)
            for (idx, data) in enumerate(batch_data):
                batch_results[idx] = qc(
                    thetas, ansatz_function, ansatz_param, pooling_function, pooling_param, data)
            loss = self.loss(batch_results, batch_labels)
            loss.backward()
            opt.step()
            if ep == 0:
                drawer = qml.draw(qc)
                print(drawer(thetas, ansatz_function, ansatz_param,
                             pooling_function, pooling_param, data))
            print('episode : ', ep, 'loss : ', loss.item())
            loss_history.append(loss.item())
        return loss_history, thetas

    def Do_Test(self, ansatz_function, ansatz_param, pooling_function, pooling_param, thetas):
        @qml.qnode(self.dev, interface='torch')
        def qc(thetas, ansatz_f, ansatz_param, pol_f, pol_param, data):
            # insert initial state as data in n_qubits, not total_qubits
            qml.QubitStateVector(data, wires=range(self.n_qubits))
            theta_idx = 0
            for layer in range(self.total_layer):
                qubit_info = np.array(self.QCNN_tree[layer])
                L = len(qubit_info)
                qubit_info_index = np.array(range(L))
                if self.stride == 1:
                    qubit_info_splited = [[idx for idx in qubit_info_index if idx % 2 == 0], [
                        idx for idx in qubit_info_index if idx % 2 == 1]]
                else:
                    qubit_info_splited = qubit_info_index
                # apply unitary
                for index in qubit_info_splited:
                    for idx in index:
                        if qubit_info[idx] != qubit_info[(idx+self.stride) % L]:
                            ansatz_f(
                                thetas[theta_idx:theta_idx+ansatz_param], wires=[qubit_info[idx], qubit_info[(idx+self.stride) % L]])
                            theta_idx += ansatz_param
                # apply pooling
                for idx in qubit_info_index:
                    if idx % 2 == 0:
                        if qubit_info[idx] != qubit_info[(idx+1) % L]:
                            pol_f(thetas[theta_idx:theta_idx+pol_param],
                                  wires=[qubit_info[idx], qubit_info[(idx+1) % L]])
                            theta_idx += pol_param
            return qml.expval(qml.PauliZ(0))
        N = 0
        N_correct = 0
        for (idx, test_data) in enumerate(self.test_qs):
            result = qc(thetas, ansatz_function, ansatz_param,
                        pooling_function, pooling_param, test_data)
            if self.test_labels[idx] == 1:
                if result > 0:
                    N_correct += 1
            elif self.test_labels[idx] == -1:
                if result < 0:
                    N_correct += 1
            else:
                print('Unable situation')
                return
            N += 1
        return N_correct/N
