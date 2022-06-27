import pennylane as qml

"""
Convolution & Pooling ansatzes used in the reference paper:
Hur, Tak, Leeseok Kim, and Daniel K. Park. "Quantum convolutional neural network for classical data classification." 
Quantum Machine Intelligence 4.1 (2022): 1-18.

To get the dimension of input weights, use the ".ndim_params" attribute to get this value.
For example:

nparams = ConvCirc1(weights = [0,1], wires = [0, 1]).ndim_params

will apply ConvCirc1 AND nparams will be equal to 2
"""

# Convolutional Layer Ansatz
class ConvCirc1(qml.operation.Operation):
    
    num_wires = qml.operation.AnyWires
    grad_method = None
    
    def __init__(self, weights = None, wires = None, do_queue=True, id=None):
        super().__init__(weights, wires=wires, do_queue=do_queue, id=id)

    @property
    def ndim_params(self):
        return 2

    @staticmethod
    def compute_decomposition(weights, wires):
        decomp = []
        decomp.append(qml.RY(weights[0], wires = wires[0]))
        decomp.append(qml.RY(weights[1], wires = wires[1]))
        decomp.append(qml.CNOT(wires = wires))
        return decomp

# Pooling Layer Ansatz
class PoolingCirc(qml.operation.Operation):

    num_wires = qml.operation.AnyWires
    grad_method = None
    
    def __init__(self, weights = None, wires = None, do_queue=True, id=None):
        super().__init__(weights, wires=wires, do_queue=do_queue, id=id)

    @property
    def ndim_params(self):
        return 2

    @staticmethod
    def compute_decomposition(weights, wires):
        decomp = []
        decomp.append(qml.CRZ(weights[0], wires = wires))
        decomp.append(qml.X(wires = wires[0]))
        decomp.append(qml.CRX(weights[1], wires = wires))
        decomp.append(qml.X(wires = wires[0]))
        return decomp