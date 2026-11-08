# IBM Qiskit Integration: Hybrid Quantum-Classical Text Classification

**University:** Sharda University  
**Course Code:** NV62010 (Quantum Computing with Artificial Intelligence)  
**Project:** Design and Implementation of a Hybrid Quantum-Classical Neural Network for Binary Text Classification Using Variational Quantum Circuits  
**Original Architecture Reference:** Huzzat et al. (IEEE Access 2026), GitHub: [AnnyshaHuzzat/Primer_QNN_Text_Classification](https://github.com/AnnyshaHuzzat/Primer_QNN_Text_Classification)

---

## 1. Overview

This module provides a complete, mathematically rigorous, independent implementation of the Variational Quantum Circuit (VQC) using **IBM Qiskit 2.6+** and **Qiskit Aer 0.18+**, alongside a custom PyTorch autograd bridge that supports end-to-end backpropagation through both classical and quantum layers.

The implementation preserves the exact mathematical semantics of the original PennyLane model while demonstrating framework-agnostic quantum computing design principles.

---

## 2. Directory Structure

```
qiskit_impl/
├── CIRCUIT_AUDIT.md            # Structural and mathematical analysis of the circuit
├── ENVIRONMENT_SETUP.md        # Environment documentation and dependencies
├── requirements-qiskit.txt     # Locked dependencies for Python 3.11/3.12
├── qiskit_circuit.py           # Core Qiskit circuit builder and Pauli-Z observables
├── run_qiskit_circuit.py       # Standalone circuit simulator and visual diagram exporter
├── verify_equivalence.py       # PennyLane vs. Qiskit numerical equivalence test suite
├── qiskit_hybrid_model.py      # Differentiable PyTorch autograd function and hybrid module
├── train_qiskit.py             # End-to-end training and evaluation script
├── benchmark_three_way.py      # Three-way comparison generator (Classical vs PennyLane vs Qiskit)
└── README.md                   # This user guide
```

Generated outputs are stored in:
- `results/qiskit/circuit_diagram.txt`: Formatted ASCII/Unicode quantum circuit
- `results/qiskit/figures/qiskit_quantum_circuit.png`: High-resolution circuit schematic
- `results/qiskit/EQUIVALENCE_REPORT.md`: Comprehensive numerical verification proof ($< 10^{-8}$ discrepancy)
- `results/qiskit/checkpoints/`: Best PyTorch model weights
- `results/qiskit/predictions/`: Test ground truth, binary predictions, and probabilities
- `results/qiskit/figures/`: Confusion matrices, training curves, ROC curves, gate count breakdown, and 3-way framework comparisons

---

## 3. Mathematical Architecture

The quantum layer maps a 4-dimensional continuous feature representation $\mathbf{x} \in \mathbb{R}^4$ (compressed from 5,000 TF-IDF features by a classical dense layer $\mathbf{W}_1 \in \mathbb{R}^{4 \times 5000}$) to 4 expectation values:

$$\langle Z_i \rangle = \langle \psi(\mathbf{x}, \boldsymbol{\theta}) | Z_i | \psi(\mathbf{x}, \boldsymbol{\theta}) \rangle, \quad i \in \{0, 1, 2, 3\}$$

### Circuit Stages:
1. **Initial State:** $|0000\rangle$
2. **Hadamard Transform:** $H^{\otimes 4}$
3. **Single-Qubit Phase Encoding:** $R_Z(x_i) = \exp(-i \frac{x_i}{2} Z_i)$
4. **Two-Qubit Entangling IQP Encoding:**
   $$R_{ZZ}(x_i x_j) = \exp\left(-i \frac{x_i x_j}{2} (Z_i \otimes Z_j)\right) \quad \text{for all } 0 \le i < j < 4$$
5. **Variational Layers ($L = 2$):**
   - For each qubit $q \in \{0,1,2,3\}$:
     $$R_X(\theta_{l, q, 0}) \to R_Y(\theta_{l, q, 1}) \to R_Z(\theta_{l, q, 2})$$
   - Linear CNOT Entangling Chain:
     $$\text{CNOT}(0 \to 1), \quad \text{CNOT}(1 \to 2), \quad \text{CNOT}(2 \to 3)$$
6. **Observables:** Expectation values of Pauli-$Z$ on each wire:
   $$O_i = Z_i \otimes I_{\text{rest}}$$

**Total Gate Count:** 44 quantum gates (32 single-qubit, 12 two-qubit entangling gates).  
**Circuit Depth:** 17.

---

## 4. Quickstart: Reproduction Commands

All commands should be executed from the repository root using the dedicated Python 3.11 environment `.venv-qiskit`:

### Step 1: Set Up Environment
```bash
python3.11 -m venv .venv-qiskit
source .venv-qiskit/bin/activate
pip install -r qiskit_impl/requirements-qiskit.txt
```

### Step 2: Render Circuit and Simulate Forward Pass
```bash
MPLCONFIGDIR=/tmp/mpl .venv-qiskit/bin/python qiskit_impl/run_qiskit_circuit.py
```
*Outputs: `results/qiskit/circuit_diagram.txt` and `results/qiskit/figures/qiskit_quantum_circuit.png`.*

### Step 3: Run PennyLane vs. Qiskit Equivalence Suite
```bash
MPLCONFIGDIR=/tmp/mpl .venv-qiskit/bin/python qiskit_impl/verify_equivalence.py
```
*Validates that $| \langle Z \rangle_{\text{PennyLane}} - \langle Z \rangle_{\text{Qiskit}} | \le 6.79 \times 10^{-9}$ across 8 test suites.*

### Step 4: Train the Hybrid Model on CR Dataset
```bash
MPLCONFIGDIR=/tmp/mpl .venv-qiskit/bin/python qiskit_impl/train_qiskit.py --dataset CR --epochs 10 --batch_size 32 --lr 0.001 --seed 42
```
*Saves model checkpoint to `results/qiskit/checkpoints/CR_qiskit_best.pt` and metrics to `results/qiskit/qiskit_model_results.csv`.*

### Step 5: Generate 3-Way Benchmark Comparison & Charts
```bash
MPLCONFIGDIR=/tmp/mpl .venv-qiskit/bin/python qiskit_impl/benchmark_three_way.py
```
*Outputs comparative charts to `results/qiskit/figures/` and tabular data to `results/qiskit/three_way_comparison.csv`.*

---

## 5. Differentiability: Parameter-Shift Rule

Because standard Qiskit circuits run through compiled DAG representations rather than pure tensor graphs, the hybrid model uses a custom `torch.autograd.Function` implementing the analytical **parameter-shift rule**:

$$\frac{\partial \langle Z \rangle}{\partial \theta} = \frac{\langle Z \rangle_{\theta + \pi/2} - \langle Z \rangle_{\theta - \pi/2}}{2}$$

For the classical encoder features $\mathbf{x} \in \mathbb{R}^4$, symmetric central difference is applied:

$$\frac{\partial \langle Z \rangle}{\partial x_i} = \frac{\langle Z \rangle_{x_i + \epsilon} - \langle Z \rangle_{x_i - \epsilon}}{2\epsilon}$$

This allows exact gradient backpropagation from PyTorch's loss function into the classical linear encoder layer $\mathbf{W}_1 \in \mathbb{R}^{4 \times 5000}$ without requiring PyTorch native C++ bindings for Qiskit.

