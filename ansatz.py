import pennylane as qml

"""
Convolution & Pooling ansatzes used in the reference paper:
Hur, Tak, Leeseok Kim, and Daniel K. Park. "Quantum convolutional neural network for classical data classification." 
Quantum Machine Intelligence 4.1 (2022): 1-18.

Make an instance of each ansatz. One can get the number of trainable parameters using the ".num_params" attribute.
Use the ".apply" method to apply the ansatz using the appropriate weights / qubits.
"""

# Convolutional Layer Ansatz
class ConvCirc1():
    
    def __init__(self) -> None:
        self.num_params = 2

    def apply(self, weights, wires):
        qml.RY(weights[0], wires = wires[0])
        qml.RY(weights[1], wires = wires[1])
        qml.CNOT(wires = wires)

# Pooling Layer Ansatz
class PoolingCirc():

    def __init__(self) -> None:
        self.num_params = 2
    
    def apply(self, weights, wires):
        qml.CRY(weights[0], wires = wires)
        qml.PauliX(wires[0])
        qml.CRY(weights[1], wires = wires)
        qml.PauliX(wires[0])

if __name__=="__main__":
    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit(w):
        ConvCirc1().apply(w, wires=(0, 1))
        return qml.expval(qml.PauliX(0))

    print(qml.draw(circuit)([0, 1]))
