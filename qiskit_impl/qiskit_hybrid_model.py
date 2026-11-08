"""
Differentiable Qiskit Quantum Layer and Hybrid PyTorch Model
============================================================
Implements a fully differentiable PyTorch autograd bridge for Qiskit:
- Computes exact forward Pauli-Z expectation values using Qiskit statevector simulation
- Computes analytical parameter-shift gradients for variational rotation weights:
    d<Z>/dw = (<Z>_{w + pi/2} - <Z>_{w - pi/2}) / 2
- Computes finite-difference / parameter-shift input gradients for the classical encoder (5000 -> 4):
    d<Z>/dx = (<Z>_{x + eps} - <Z>_{x - eps}) / (2 * eps)
- Supports full end-to-end backpropagation through:
    Loss -> fc2 -> Qiskit Quantum Layer -> fc1 (encoder) -> TF-IDF inputs
- Utilizes batched statevector / unitary caching for high numerical efficiency
"""

import torch
import torch.nn as nn
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from qiskit_impl.qiskit_circuit import build_qiskit_circuit, compute_pauli_z_expectations

def _get_signs_table(n_qubits=4):
    k = np.arange(1 << n_qubits)
    return np.array([1.0 - 2.0 * ((k >> i) & 1) for i in range(n_qubits)], dtype=np.float32)

def _build_feature_states(x_batch, n_qubits=4):
    B = x_batch.shape[0]
    sv_mat = np.zeros((1 << n_qubits, B), dtype=np.complex128)
    for b in range(B):
        qc = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            qc.h(i)
        for i in range(n_qubits):
            qc.rz(x_batch[b, i], i)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qc.rzz(x_batch[b, i] * x_batch[b, j], i, j)
        sv_mat[:, b] = Statevector.from_instruction(qc).data
    return sv_mat

def _build_var_unitary(w, n_qubits=4, n_layers=2):
    qc = QuantumCircuit(n_qubits)
    for l in range(n_layers):
        for q in range(n_qubits):
            qc.rx(w[l, q, 0], q)
            qc.ry(w[l, q, 1], q)
            qc.rz(w[l, q, 2], q)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    return Operator(qc).data

class QiskitQuantumFunction(torch.autograd.Function):
    """
    Custom PyTorch autograd Function that evaluates Qiskit circuits
    and propagates exact gradients to both inputs and variational weights.
    """

    @staticmethod
    def forward(ctx, inputs, weights, n_qubits=4, n_layers=2):
        ctx.n_qubits = n_qubits
        ctx.n_layers = n_layers
        
        inputs_np = inputs.detach().cpu().numpy().astype(np.float32)
        weights_np = weights.detach().cpu().numpy().astype(np.float32)
        
        batch_size = inputs_np.shape[0]
        signs = _get_signs_table(n_qubits)
        
        # Batched forward simulation
        sv_feat = _build_feature_states(inputs_np, n_qubits=n_qubits)
        U_var = _build_var_unitary(weights_np, n_qubits=n_qubits, n_layers=n_layers)
        probs = np.abs(U_var @ sv_feat) ** 2
        outputs = (signs @ probs).T.astype(np.float32)

        ctx.save_for_backward(inputs, weights)
        ctx.sv_feat = sv_feat
        ctx.U_var = U_var
        ctx.signs = signs
        
        return torch.tensor(outputs, dtype=inputs.dtype, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weights = ctx.saved_tensors
        n_qubits = ctx.n_qubits
        n_layers = ctx.n_layers
        sv_feat = ctx.sv_feat
        U_var = ctx.U_var
        signs = ctx.signs

        inputs_np = inputs.detach().cpu().numpy().astype(np.float32)
        weights_np = weights.detach().cpu().numpy().astype(np.float32)
        grad_output_np = grad_output.detach().cpu().numpy().astype(np.float32)
        batch_size = inputs_np.shape[0]

        grad_inputs_np = np.zeros_like(inputs_np)
        grad_weights_np = np.zeros_like(weights_np)

        # 1. Parameter-shift rule for variational weights (shift s = pi / 2)
        shift_w = np.pi / 2.0
        for l in range(n_layers):
            for q in range(n_qubits):
                for r in range(3):
                    w_pos = weights_np.copy(); w_pos[l, q, r] += shift_w
                    w_neg = weights_np.copy(); w_neg[l, q, r] -= shift_w
                    U_pos = _build_var_unitary(w_pos, n_qubits=n_qubits, n_layers=n_layers)
                    U_neg = _build_var_unitary(w_neg, n_qubits=n_qubits, n_layers=n_layers)

                    exp_pos = (signs @ (np.abs(U_pos @ sv_feat) ** 2)).T
                    exp_neg = (signs @ (np.abs(U_neg @ sv_feat) ** 2)).T
                    d_exp_dw = (exp_pos - exp_neg) / 2.0

                    grad_weights_np[l, q, r] = np.sum(grad_output_np * d_exp_dw)

        # 2. Gradient w.r.t inputs (to train classical encoder fc1)
        # Central difference taking all RZ and RZZ interactions into account
        eps_x = 1e-3
        for q in range(n_qubits):
            x_pos = inputs_np.copy(); x_pos[:, q] += eps_x
            x_neg = inputs_np.copy(); x_neg[:, q] -= eps_x
            sv_feat_pos = _build_feature_states(x_pos, n_qubits=n_qubits)
            sv_feat_neg = _build_feature_states(x_neg, n_qubits=n_qubits)

            exp_x_pos = (signs @ (np.abs(U_var @ sv_feat_pos) ** 2)).T
            exp_x_neg = (signs @ (np.abs(U_var @ sv_feat_neg) ** 2)).T
            d_exp_dx = (exp_x_pos - exp_x_neg) / (2.0 * eps_x)

            grad_inputs_np[:, q] = np.sum(grad_output_np * d_exp_dx, axis=1)

        grad_inputs = torch.tensor(grad_inputs_np, dtype=inputs.dtype, device=inputs.device)
        grad_weights = torch.tensor(grad_weights_np, dtype=weights.dtype, device=weights.device)

        return grad_inputs, grad_weights, None, None


class QiskitTorchLayer(nn.Module):
    """
    PyTorch nn.Module wrapper around QiskitQuantumFunction.
    Exposes trainable parameters of shape (n_layers, n_qubits, 3).
    """
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        initial_weights = np.random.uniform(-np.pi, np.pi, size=(n_layers, n_qubits, 3)).astype(np.float32)
        self.weights = nn.Parameter(torch.tensor(initial_weights, requires_grad=True))

    def forward(self, x):
        return QiskitQuantumFunction.apply(x, self.weights, self.n_qubits, self.n_layers)


class QiskitHybridClassifier(nn.Module):
    """
    End-to-end Hybrid Quantum-Classical Text Classifier using IBM Qiskit:
    1. Classical linear feature compressor: 5000 -> 4
    2. Non-linear activation: tanh (maps to [-1, 1] suitable for angle embedding)
    3. Differentiable Qiskit VQC Layer: 4 -> 4
    4. Classical readout linear layer: 4 -> 2
    """
    def __init__(self, input_dim=5000, n_qubits=4, n_layers=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, n_qubits)
        self.qlayer = QiskitTorchLayer(n_qubits=n_qubits, n_layers=n_layers)
        self.fc2 = nn.Linear(n_qubits, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.qlayer(x)
        x = self.fc2(x)
        return x
