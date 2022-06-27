import torch as T
import numpy as np
import MNIST_to_Quantum as MQ


class QCNN:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.current_layer = 0
        # construct QCNN tree
        self.total_layer = 0
        # QCNN tree location
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

    # ansatz info : information about building ansatz
    # ansatz 1,2,....,N
    # ansatz 0 : general unitary

    def Get_Unitary(self, ansatz_infos, ansatz_num):
        # total unitary
        total_U = T.eye(2**self.QCNN_tree[0], dtype=T.cfloat)
        # apply unitary for each layer
        for layer in range(self.total_layer):
            layer_ansatz_info = ansatz_infos[layer]
            for ansatz in layer_ansatz_info:
                # qml.??
                return
            return
        return

    def partial_trace_out(self):
        return

    def Do_QCNN(self):
        return


A = QCNN(5)
print(A.QCNN_tree)
print(A.ancilla_info)
