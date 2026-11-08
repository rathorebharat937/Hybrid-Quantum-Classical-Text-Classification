# Implementation Report: Hybrid Quantum-Classical Text Classification

**Project:** Quantum Neural Network for Text Classification Using Hybrid Quantum-Classical Learning  
**Author / Student:** Final Year B.Tech Project  
**Date:** November 2026  
**Hardware & System:** Apple M-series (arm64, 8 cores, 8 GB RAM), macOS 15.5  
**Core Frameworks:** Python 3.14.6, PennyLane 0.45.1, PyTorch 2.15.0, Scikit-Learn 1.9.1  

---

## 1. Technical Audit & Code Debugging Summary

During Phase 1 and Phase 2 inspection, several functional issues and blockers were identified and systematically resolved:

1. **Parameter Mismatch in Model Construction:**
   - *Problem:* `src/train.py` passed `n_layers=cfg["n_layers"]` into `HybridClassifier`, but `HybridClassifier.__init__()` in `src/model.py` only accepted `input_dim` and `n_qubits`.
   - *Fix:* Added `n_layers=2` to `HybridClassifier.__init__()` and forwarded it directly to `build_qlayer(n_qubits=n_qubits, n_layers=n_layers)`.

2. **Hardcoded Circuit Dimensions in PennyLane Interface:**
   - *Problem:* `src/circuit.py` declared global module-level variables `n_qubits = 4` and `n_layers = 2`, preventing flexible parameterization and ablation studies.
   - *Fix:* Modularized `build_qlayer(n_qubits=4, n_layers=2)` to instantiate the quantum device (`qml.device("default.qubit", wires=n_qubits)`) and torch layer dynamically with arbitrary qubit count and layer depth.

3. **Missing Local Datasets & Placeholder Paths:**
   - *Problem:* `src/data.py` contained hardcoded strings `"Enter the Dataset Location"` for MR, CR, and SST.
   - *Fix:* Developed `download_datasets.py` to auto-fetch the canonical benchmark versions of MR (`cornell-movie-review-data/rotten_tomatoes`), CR (`SetFit/cr`), and SST (`SetFit/sst2`), extracting and staging them into `data/mr/`, `data/cr/`, and `data/sst/`. Configured `src/data.py` with relative paths matching the exact original file structures.

4. **Missing Classical Baseline Models & Ablation Harness:**
   - *Problem:* The repository lacked classical comparison baselines (Logistic Regression, Linear SVM, Classical Deep MLP, and dimension-matched Compact MLP) as well as automated ablation tracking.
   - *Fix:* Created `run_experiments.py` executing 25 primary model benchmark runs (5 datasets × 5 model types) plus 9 controlled quantum ablation experiments.

---

## 2. Experimental Pipeline Architecture

```
+-------------------------------------------------------------+
|                      Raw Text Input                         |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|             TF-IDF Preprocessing (dim = 5000)               |
+-------------------------------------------------------------+
                              |
               +--------------+--------------+
               |                             |
               v                             v
+-----------------------------+ +-----------------------------+
| Classical Baselines         | | Hybrid QNN Pipeline         |
| 1. Logistic Regression      | | 1. Linear: 5000 -> 4        |
| 2. Calibrated Linear SVM    | | 2. Activation: tanh         |
| 3. Deep MLP (5000->64->32->2| | 3. IQP Embedding (4 qubits) |
| 4. Compact MLP (5000->4->2) | | 4. VQC (RX, RY, RZ + CNOT)  |
|                             | | 5. Pauli-Z Measurements (<Z>|
|                             | | 6. Linear: 4 -> 2 Output    |
+-----------------------------+ +-----------------------------+
               |                             |
               +--------------+--------------+
                              |
                              v
+-------------------------------------------------------------+
| Held-out Test Set Evaluation (Acc, Prec, Rec, F1, AUC, Time)|
+-------------------------------------------------------------+
```

---

## 3. Quantum Circuit Design & Differentiability

- **Simulator Device:** `default.qubit` statevector simulator integrated via PennyLane's TorchLayer interface.
- **State Encoding:** IQP (Instantaneous Quantum Polynomial) Embedding, introducing non-linear angle interactions between input features across the quantum register.
- **Variational Ansatz:** Strongly entangling architecture using parameterized single-qubit rotations ($R_x(\theta_1), R_y(\theta_2), R_z(\theta_3)$) followed by linear nearest-neighbor CNOT entangling gates.
- **Measurements:** Expectation values of Pauli-Z operators $\langle \sigma_z^{(i)} \rangle$ for each qubit $i \in \{0, \dots, n-1\}$, yielding bounded observables in $[-1, 1]$.
- **Autograd Compatibility:** Fully end-to-end backpropagation verified through PyTorch Adam optimizer. Gradients propagate seamlessly from Cross-Entropy loss through the classical linear layers and quantum circuit parameters.

---

## 4. Checkpoints and Artifacts

- Model checkpoints saved to `checkpoints/`:
  - `*_QNN_best.pt` (Hybrid QNN best validation epoch weights)
  - `*_MLP_best.pt` (Classical MLP best validation epoch weights)
- Full predictions stored in `results/predictions/*.npz` containing arrays `y_true`, `y_pred`, and calibrated probabilities `y_prob`.
- Comprehensive metrics output to `results/model_comparison.csv` and `results/experimental_summary.json`.

