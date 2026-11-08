# Quantum Circuit Audit: PennyLane vs Qiskit Mapping

**Date:** November 2026  
**Project:** Hybrid Quantum-Classical Text Classification  
**Source Circuit File:** `src/circuit.py`  

---

## 1. PennyLane Quantum Circuit Analysis

In `src/circuit.py`, the quantum model is constructed via:
```python
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
```

### 1.1 Structural Properties
1. **Qubit Count ($n$):** Configurable, default $n = 4$ qubits (wires $0, 1, 2, 3$).
2. **Feature Embedding:** `qml.IQPEmbedding(inputs, wires=range(n_qubits))`.
   - Inspection of PennyLane's internal implementation (`qml.IQPEmbedding.compute_decomposition`) shows:
     - Applies Hadamard gate $H$ to each wire $i \in \{0, \dots, n-1\}$.
     - Applies single-qubit rotation $R_Z(x_i)$ to wire $i$.
     - Applies two-qubit interaction for all unique pairs $(i, j)$ with $i < j$:
       `qml.MultiRZ(x_i * x_j, wires=[i, j])`.
     - `MultiRZ(\theta, wires=[i, j])` decomposes into:
       `CNOT(j, i) -> R_Z(\theta, i) -> CNOT(j, i)`.
       (Alternatively `CNOT(i, j) -> R_Z(\theta, j) -> CNOT(i, j)` or native Qiskit `rzz(theta, i, j)`).
3. **Variational Layers ($L$):** Configurable, default $L = 2$ layers.
   - Per layer $l \in \{0, \dots, L-1\}$ and for each wire $i \in \{0, \dots, n-1\}$:
     - Single-qubit rotation: $R_X(\theta_{l, i, 0})$.
     - Single-qubit rotation: $R_Y(\theta_{l, i, 1})$.
     - Single-qubit rotation: $R_Z(\theta_{l, i, 2})$.
   - Linear nearest-neighbor entanglement:
     - `CNOT(0, 1)`, `CNOT(1, 2)`, ..., `CNOT(n-2, n-1)`.
     - Total $n-1$ CNOT gates per variational layer ($3$ CNOTs for $n=4$).
4. **Trainable Parameter Tensor:**
   - Shape: $(L, n, 3) = (2, 4, 3) = 24$ continuous parameters.
5. **Observables / Measurements:**
   - Expectation values of Pauli-$Z$ on each individual qubit:
     $\langle Z_i \rangle = \langle \psi | Z_i | \psi \rangle \in [-1, 1]$ for $i \in \{0, 1, 2, 3\}$.
   - Output dimension: $\mathbb{R}^4$.

---

## 2. Qiskit Conversion Strategy & Architectural Alignment

### 2.1 Qubit Indexing and Pauli Conventions
- **PennyLane vs Qiskit Qubit Ordering:**
  - PennyLane uses big-endian wire ordering in statevectors where wire 0 is the most significant bit: $|w_0 w_1 w_2 w_3\rangle$.
  - Qiskit uses little-endian ordering where qubit 0 is the least significant bit: $|q_3 q_2 q_1 q_0\rangle$.
  - However, for **single-qubit expectation values $\langle Z_i \rangle$**, wire $i$ in PennyLane corresponds strictly to qubit $i$ in Qiskit:
    $\langle \psi | I \dots Z_i \dots I | \psi \rangle$.
  - In Qiskit `SparsePauliOp`, string positions are indexed from right to left: qubit 0 is the rightmost character. E.g., for 4 qubits:
    - Qubit 0: `'IIIZ'`
    - Qubit 1: `'IIZI'`
    - Qubit 2: `'IZII'`
    - Qubit 3: `'ZIII'`

### 2.2 Gate Equivalences
1. **Hadamard Gate:** PennyLane `H(i)` $\equiv$ Qiskit `qc.h(i)`.
2. **Rotations:**
   - PennyLane $R_X(\theta), R_Y(\theta), R_Z(\theta)$ use standard definitions:
     $R_X(\theta) = \exp(-i \frac{\theta}{2} X)$, etc.
   - Qiskit `qc.rx(theta, i)`, `qc.ry(theta, i)`, `qc.rz(theta, i)` use the identical mathematical convention.
3. **MultiRZ / Entangling Embeddings:**
   - PennyLane's `MultiRZ(x_i * x_j, wires=[i, j])` applies unitary $\exp(-i \frac{x_i x_j}{2} Z_i Z_j)$.
   - Qiskit's `qc.rzz(theta, i, j)` applies unitary $\exp(-i \frac{\theta}{2} Z_i \otimes Z_j)$.
   - Setting $\theta = x_i \cdot x_j$ in Qiskit's `rzz(x_i * x_j, i, j)` provides exact mathematical equivalence.
4. **Entangling Ladder:**
   - PennyLane `qml.CNOT(wires=[i, i+1])` $\equiv$ Qiskit `qc.cx(i, i+1)`.

---

## 3. Implementation Challenges & Solutions

| Challenge | Impact | Qiskit Resolution Strategy |
| :--- | :--- | :--- |
| **Qubit Endianness** | Statevector ordering differs | Construct explicit `SparsePauliOp` operators with proper qubit indexing |
| **Differentiability with PyTorch** | Qiskit circuits do not natively track PyTorch computational graphs | Implement a custom `torch.autograd.Function` utilizing analytical parameter-shift gradients or fast statevector autograd |
| **Batch Evaluation** | PyTorch passes mini-batches of size 32 | Vectorize statevector expectation value computations across batch dimensions for maximum CPU throughput |

