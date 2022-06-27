import torch as T
import numpy as np
import MNIST_to_Quantum as MQ
import pennylane as qml


class QCNN:
    def __init__(self, n_qubits):
        self.MQ = MQ(16)
        self.n_qubits = n_qubits
        self.dev = qml.deivce('default.qubit', wires=self.n_qubits)
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
        # generate MNIST dataset
        # train set : 1000, test set : 100
        self.MQ.GET_MNIST(1000, 100, [0, 1])  # label 0,1
        self.train_qs = self.MQ.train_quantum_states
        self.test_qs = self.MQ.test_quantum_states

    # ansatz info : information about building ansatz
    # ansatz 1,2,....,N
    # ansatz 0 : general unitary

    def Get_Unitary(self, ansatz_infos, total_param_num):
        # theta initialization
        thetas = T.tensor(np.random.rand(total_param_num),
                          dtype=T.float, requires_grad=True)
        # theta index
        theta_idx = 0
        # apply unitary for each layer
        for layer in range(self.total_layer):
            # layer ansatz info : set of gate informations
            layer_ansatz_info = ansatz_infos[layer]
            # ansatz info : [gatetype,control_qubit,target_qubit]
            for ansatz in layer_ansatz_info:
                gt = ansatz[0]
                cq = ansatz[1]
                tq = ansatz[2]
                # Hadamard
                if gt == 0:
                    qml.Hadamard()
                # RX
                elif gt == 1:
                    qml.RX(thetas[theta_idx], wires=[tq])
                    theta_idx += 1
                # RY
                elif gt == 2:
                    qml.RY(thetas[theta_idx], wires=[tq])
                    theta_idx += 1
                # RZ
                elif gt == 3:
                    qml.RZ(thetas[theta_idx], wires=[tq])
                    theta_idx += 1
                # CNOT
                elif gt == 4:
                    qml.CNOT(wires=[cq, tq])
                # CZ
                elif gt == 5:
                    qml.CZ(wires=[cq, tq])
                # CRX
                elif gt==6:
                    qml.CRX(thetas[theta_idx],wires=[cq,tq])
                    theta_idx+=1
                # CRY
                elif gt==7:
                    qml.CRY(thetas[theta_idx],wires=[cq,tq])
                    theta_idx+=1
                # CRZ
                elif gt==8:
                    qml.CRZ(thetas[theta_idx],wires=[cq,tq])
                    theta_idx+=1

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
