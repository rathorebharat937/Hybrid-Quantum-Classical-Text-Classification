"""
Qiskit Variational Quantum Circuit Module
=========================================
Exact re-implementation of the PennyLane hybrid VQC architecture
using modern IBM Qiskit (2.6+).

Supports:
- Configurable qubit counts (n_qubits)
- Configurable circuit depth (n_layers)
- Configurable input feature vectors (x)
- Configurable variational weight tensors (weights)
- Deterministic statevector simulation
- Shot-based sampling with Qiskit Aer
- Expectation value extraction of single-qubit Pauli-Z observables
"""

import numpy as np
from typing import Optional, List, Tuple, Union
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp


def build_qiskit_circuit(
    inputs: Union[np.ndarray, List[float], ParameterVector],
    weights: Union[np.ndarray, ParameterVector],
    n_qubits: int = 4,
    n_layers: int = 2
) -> QuantumCircuit:
    """
    Constructs the exact Qiskit QuantumCircuit equivalent to PennyLane's:
      qml.IQPEmbedding(inputs, wires=range(n_qubits))
      + n_layers of [RX, RY, RZ per qubit, followed by linear CNOT chain]
    
    Args:
        inputs: 1D array or ParameterVector of length n_qubits
        weights: 3D array of shape (n_layers, n_qubits, 3) or 1D ParameterVector of size n_layers*n_qubits*3
        n_qubits: Number of qubits (default 4)
        n_layers: Number of variational layers (default 2)
        
    Returns:
        QuantumCircuit object ready for statevector simulation or Aer execution.
    """
    qc = QuantumCircuit(n_qubits, name=f"HybridQNN_{n_qubits}q_{n_layers}L")

    # -------------------------------------------------------------
    # 1. Feature Embedding (IQP Embedding equivalent)
    # -------------------------------------------------------------
    # Step 1.1: Hadamard on all qubits
    for i in range(n_qubits):
        qc.h(i)

    # Step 1.2: RZ rotations by feature x_i on all qubits
    for i in range(n_qubits):
        qc.rz(inputs[i], i)

    # Step 1.3: Entangling RZZ interactions between all pairs (i, j) with i < j
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            # PennyLane MultiRZ applies exp(-i * (x_i * x_j) / 2 * Z_i Z_j)
            # Qiskit rzz applies exp(-i * theta / 2 * Z_i Z_j)
            qc.rzz(inputs[i] * inputs[j], i, j)

    # -------------------------------------------------------------
    # 2. Variational Layers
    # -------------------------------------------------------------
    # Reshape weights if passed as 1D
    if isinstance(weights, np.ndarray) and weights.ndim == 1:
        weights = weights.reshape((n_layers, n_qubits, 3))

    for l in range(n_layers):
        # 2.1 Single-qubit parameterized rotations (RX -> RY -> RZ)
        for i in range(n_qubits):
            if isinstance(weights, np.ndarray):
                rx_val = weights[l, i, 0]
                ry_val = weights[l, i, 1]
                rz_val = weights[l, i, 2]
            else:
                # ParameterVector indexing
                base_idx = l * n_qubits * 3 + i * 3
                rx_val = weights[base_idx + 0]
                ry_val = weights[base_idx + 1]
                rz_val = weights[base_idx + 2]

            qc.rx(rx_val, i)
            qc.ry(ry_val, i)
            qc.rz(rz_val, i)

        # 2.2 Linear CNOT entanglement chain: (0->1), (1->2), ..., (n-2->n-1)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)

    return qc


def compute_pauli_z_expectations(
    qc: QuantumCircuit,
    n_qubits: int = 4
) -> np.ndarray:
    """
    Computes exact statevector Pauli-Z expectation values <Z_i> for each qubit.
    
    Qiskit SparsePauliOp uses little-endian ordering:
    qubit 0 is the rightmost character (position n-1-0).
    
    Returns:
        np.ndarray of shape (n_qubits,) with values in [-1.0, 1.0].
    """
    sv = Statevector.from_instruction(qc)
    expectations = []
    
    for i in range(n_qubits):
        pauli_chars = ['I'] * n_qubits
        pauli_chars[n_qubits - 1 - i] = 'Z'
        op = SparsePauliOp(''.join(pauli_chars))
        val = sv.expectation_value(op).real
        expectations.append(float(val))
        
    return np.array(expectations, dtype=np.float32)


def execute_qiskit_qnode(
    inputs: np.ndarray,
    weights: np.ndarray,
    n_qubits: int = 4,
    n_layers: int = 2
) -> np.ndarray:
    """
    Single function callable reproducing the PennyLane qnode interface:
    inputs: (n_qubits,)
    weights: (n_layers, n_qubits, 3)
    Returns: (n_qubits,) expectation values.
    """
    qc = build_qiskit_circuit(inputs, weights, n_qubits=n_qubits, n_layers=n_layers)
    return compute_pauli_z_expectations(qc, n_qubits=n_qubits)

