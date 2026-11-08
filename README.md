# Hybrid Quantum-Classical Neural Network for Binary Text Classification

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.44%2B-orange.svg)](https://pennylane.ai/)
[![IBM Qiskit](https://img.shields.io/badge/IBM_Qiskit-2.6.0-6929C4.svg)](https://qiskit.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Academic Institution:** Sharda University  
**Course:** NV62010 — Quantum Computing with Artificial Intelligence  
**Author:** Bharat Rathore ([@rathorebharat937](https://github.com/rathorebharat937))  
**Scientific Reference:** Based on *Huzzat et al., IEEE Access 2026*

---

## 📌 Executive Summary

This repository presents the design, implementation, mathematical verification, and comparative benchmarking of a **Hybrid Quantum-Classical Neural Network (HQNN)** for binary text classification. 

Initially introduced using PennyLane, this project completes a **full IBM Qiskit integration**, establishing dual-framework quantum circuit execution, custom analytical parameter-shift gradient backpropagation in PyTorch, algebraic equivalence proofs, and a three-way performance benchmark against classical baselines.

```
Raw Text ──► [ TF-IDF Vectorizer ] ──► ℝ^5000 / ℝ^4760
                    │
                    ▼
             [ Linear Encoder + Tanh ] ──► ℝ^4 (Latent Features)
                    │
                    ▼
     ┌────────────────────────────────────────────────────────┐
     │      4-Qubit Variational Quantum Circuit (VQC)         │
     │  - Angle Feature Encoding: R_x(x_i)                    │
     │  - L=2 Entangling Layers: [Rx, Ry, Rz] + Circular CNOT │
     │  - Expectation Measurement: ⟨Z_i⟩ across all 4 qubits  │
     │    Implemented in BOTH PennyLane and IBM Qiskit        │
     └────────────────────────────────────────────────────────┘
                    │
                    ▼  ⟨Z⟩ ∈ [-1, 1]^4
             [ Linear Classifier ] ──► Logits [z_0, z_1] ──► Class Prediction
```

---

## 🌟 Key Contributions & Features

1. **Dual Quantum Engine Architecture:**
   - **PennyLane Engine (`src/`):** Baseline implementation using `default.qubit` and `qml.qnn.TorchLayer`.
   - **IBM Qiskit Engine (`qiskit_impl/`):** Full end-to-end implementation with native `qiskit.circuit.QuantumCircuit` parameterized binding and statevector execution (`qiskit.quantum_info.Statevector`).
2. **Custom PyTorch Parameter-Shift Autograd Function (`qiskit_impl/qiskit_hybrid_model.py`):**
   - Implements the exact analytical parameter-shift rule:
     $$\frac{\partial \langle Z \rangle}{\partial \theta} = \frac{1}{2} \left[ \langle Z \rangle_{\theta + \pi/2} - \langle Z \rangle_{\theta - \pi/2} \right]$$
   - Algorithmic acceleration: batches unitary operator transformations across samples, slashing epoch latency by **10.2×**.
3. **Rigorous Numerical Equivalence:**
   - Validated over 10 deterministic test suites: quantum statevector fidelity $F = 1.0000000000$ ($1 - F < 1.11 \times 10^{-15}$), maximum expectation deviation $< 2.96 \times 10^{-8}$.
4. **Gradient Verification:**
   - Variational parameter gradients verified against centered finite differences with error $< 5.06 \times 10^{-9}$.
5. **Three-Way Empirical Benchmark:**
   - Evaluated across Classical Compact MLP (4-dim bottleneck), PennyLane Hybrid QNN, and IBM Qiskit Hybrid QNN.

---

## 📊 Experimental Results (Customer Reviews - CR Dataset)

Full evaluation on the CR test set ($N = 384$ test samples, 3,024 train samples):

| Metric | Classical Compact MLP | PennyLane Hybrid QNN | IBM Qiskit Hybrid QNN |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 79.43% | **80.47%** | **79.95%** |
| **Weighted F1** | 79.37% | **80.37%** | **79.90%** |
| **Macro F1** | 77.47% | **78.47%** | **78.19%** |
| **Positive Class F1** | 84.05% | **84.85%** | **84.38%** |
| **ROC-AUC** | 87.11% | **87.52%** | **87.43%** |
| **Train Time (10 Epochs)** | **0.33 s** | 6.06 s | 95.78 s |
| **Inference Time (Eval)** | **0.003 s** | 0.081 s | 0.093 s |
| **Trainable Parameters** | 19,078 | 19,078 | 19,078 |

> **Key Scientific Finding:** The Classical Compact MLP with the exact same 4-dimensional bottleneck achieves 79.43% accuracy—within 0.52% of the Quantum model—while executing 290× faster. Classification capacity is predominantly driven by the classical linear projection layer ($4,760 \to 4$). At 4 qubits in the NISQ regime, simulation exhibits parity with classical bottlenecks without empirical quantum advantage.

---

## 📁 Repository Structure

```
.
├── src/                               # Original PennyLane Implementation
│   ├── circuit.py                     # 4-qubit VQC (qml.AngleEmbedding + Rotations + CNOTs)
│   ├── model.py                       # Hybrid QNN with TorchLayer
│   ├── data.py                        # TF-IDF vectorization and data loaders
│   └── train.py                       # PennyLane training pipeline
├── qiskit_impl/                       # IBM Qiskit Implementation
│   ├── qiskit_circuit.py              # Native Qiskit QuantumCircuit & drawing
│   ├── qiskit_hybrid_model.py         # PyTorch Autograd with parameter-shift & caching
│   ├── train_qiskit.py                # Qiskit model training script
│   ├── verify_equivalence.py          # 10-suite PennyLane vs Qiskit verification
│   ├── validate_gradients_comprehensive.py # Gradient parameter-shift validation
│   ├── benchmark_three_way.py         # Three-way model comparative evaluation
│   ├── audit_runtime.py               # Runtime latency profiling
│   └── requirements-qiskit.txt        # Qiskit virtual environment dependencies
├── results/                           # Experimental Data & Checkpoints
│   ├── qiskit/                        # Qiskit-specific results
│   │   ├── checkpoints/CR_qiskit_best.pt   # Best trained Qiskit model checkpoint
│   │   ├── figures/                   # Confusion matrices, ROC curves, circuit diagrams
│   │   ├── three_way_comparison.csv   # Unified benchmarking metrics
│   │   └── qiskit_training_log.csv    # Epoch-by-epoch loss & metric trajectory
│   ├── model_comparison.csv           # Baseline multi-dataset metrics
│   └── repeated_seed_summary.csv      # Statistical validation across seeds
├── data/                              # Benchmark Datasets (CR, MR, SST)
├── COMPLETE_QISKIT_PROJECT_REPORT.md  # Comprehensive 46 KB Master Technical Report
├── QISKIT_FINAL_VERIFICATION.md       # Scientific audit & verification matrix
├── FINAL_PROJECT_REPORT.md            # Complete final-year university report
└── Qiskit_Integration_Package.zip     # Portable standalone submission archive
```

---

## 🛠️ Installation & Setup

### Option 1: PennyLane Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Option 2: IBM Qiskit Environment (Recommended for Qiskit modules)
```bash
python3.11 -m venv .venv-qiskit
source .venv-qiskit/bin/activate
pip install -r qiskit_impl/requirements-qiskit.txt
```

---

## 🚀 Usage & Reproduction

### 1. Verify Mathematical Equivalence (PennyLane vs. Qiskit)
Runs 10 randomized suites comparing statevectors, observables, and gate sequences:
```bash
python -m qiskit_impl.verify_equivalence
```
*Expected: `All 10 verification suites PASSED. Max observable deviation < 1e-7.`*

### 2. Validate Parameter-Shift Gradients
Verifies analytical quantum gradients against centered finite differences:
```bash
python -m qiskit_impl.validate_gradients_comprehensive
```
*Expected: `All gradient tests PASSED. Max error < 1e-5.`*

### 3. Train the Qiskit Hybrid Model
Train the full hybrid model on the Customer Reviews (CR) dataset:
```bash
python -m qiskit_impl.train_qiskit --dataset CR --epochs 10 --batch_size 32 --lr 0.001
```

### 4. Run the Three-Way Unified Benchmark
Evaluate Classical Compact MLP, PennyLane, and Qiskit side-by-side:
```bash
python -m qiskit_impl.benchmark_three_way --dataset CR
```

---

## 🔬 Quantum Circuit Specification

- **Qubits ($N$):** 4
- **State Space:** Hilbert space dimension $\mathcal{H} = 2^4 = 16$
- **Feature Encoding:** Angle embedding using $R_x(x_i)$ on each qubit $i \in \{0, 1, 2, 3\}$
- **Variational Layers ($L$):** 2
- **Rotations per Qubit per Layer:** $R_x(\theta) \to R_y(\theta) \to R_z(\theta)$ ($3 \times 4 = 12$ angles/layer)
- **Entanglement:** Circular nearest-neighbor CNOT chain: `(0→1)`, `(1→2)`, `(2→3)`, `(3→0)`
- **Total Quantum Angles:** $L \times N \times 3 = 2 \times 4 \times 3 = \mathbf{24\text{ parameters}}$
- **Observables:** Longitudinal Pauli-Z expectation values $\langle Z_i \rangle = \langle \psi | Z_i | \psi \rangle \in [-1, 1]$
- **Circuit Count per Training Sample:** 57 circuits ($1 \text{ forward} + 48 \text{ weight shifts} + 8 \text{ feature shifts}$)

```
q_0: ──Rx(x0)──Rx(θ0)──Ry(θ1)──Rz(θ2)──■───────────────X──...──M(Z)
                                       │               │
q_1: ──Rx(x1)──Rx(θ3)──Ry(θ4)──Rz(θ5)──X──────■────────┼──...──M(Z)
                                              │        │
q_2: ──Rx(x2)──Rx(θ6)──Ry(θ7)──Rz(θ8)─────────X────■───┼──...──M(Z)
                                                   │   │
q_3: ──Rx(x3)──Rx(θ9)──Ry(θ10)─Rz(θ11)─────────────X───■──...──M(Z)
```

---

## 📜 Academic Integrity & Citation

This research was conducted as part of the final-year B.Tech curriculum at Sharda University. All experiments adhere to open-science reproducibility standards:
- Local statevector simulations were executed via `qiskit.quantum_info.Statevector` and `default.qubit` (no physical quantum hardware execution is claimed).
- The baseline paper reference:
```bibtex
@article{huzzat2026primer,
  title={A Primer on Quantum Neural Network Model for Text Classification: From Concept to Design},
  author={Huzzat, Annysha and Khwaja, Ahmed S and Adhikari, Bhagawat and Alnoman, Ali and Woungang, Isaac and Anpalagan, Alagan},
  journal={IEEE Access},
  year={2026},
  publisher={IEEE}
}
```

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
