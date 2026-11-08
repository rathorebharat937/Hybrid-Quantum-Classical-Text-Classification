import pennylane as qml
import torch.nn as nn


def build_qlayer(n_qubits=4, n_layers=2):
    dev = qml.device("default.qubit", wires=n_qubits)

    def variational_layer(weights, n_q):
        for i in range(n_q):
            qml.RX(weights[i, 0], wires=i)
            qml.RY(weights[i, 1], wires=i)
            qml.RZ(weights[i, 2], wires=i)
        for i in range(n_q - 1):
            qml.CNOT(wires=[i, i + 1])

    @qml.qnode(dev, interface="torch")
    def qnode(inputs, weights):
        qml.IQPEmbedding(inputs, wires=range(n_qubits))
        for w in weights:
            variational_layer(w, n_qubits)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    return qml.qnn.TorchLayer(qnode, weight_shapes)
