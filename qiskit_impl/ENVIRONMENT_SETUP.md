# Qiskit Environment Setup Documentation

**Environment Path:** `.venv-qiskit/`  
**Base Python:** Python 3.11.15 (`/opt/homebrew/bin/python3.11`)  
**Host System:** Apple Silicon (arm64), macOS 15.5 Sequoia  
**Installation Date:** November 2026  

---

## 1. Environment Motivation & Isolation

While the base project virtual environment (`venv/`) utilizes Python 3.14.6 for the original PennyLane experiments, `qiskit` and `qiskit-aer` currently offer official stable wheels and precompiled C++ Aer simulation extensions for Python 3.9 through 3.12.

To guarantee zero interference with existing PennyLane models, checkpoints, and benchmark results, a dedicated isolated virtual environment was instantiated at `.venv-qiskit/` using Python 3.11.

---

## 2. Installed Software Versions

| Package | Version | Purpose |
| :--- | :--- | :--- |
| **`qiskit`** | **2.6.2** | Core quantum circuit construction and operator manipulation |
| **`qiskit-aer`** | **0.18.2** | High-performance C++ statevector and density matrix simulation |
| **`torch`** | **2.15.0** | PyTorch machine learning framework and automatic differentiation |
| **`pennylane`** | **0.45.1** | Cross-framework numerical equivalence verification |
| **`numpy`** | **2.4.6** | Mathematical array operations |
| **`scipy`** | **1.17.1** | Scientific computing and optimization |
| **`scikit-learn`**| **1.9.1** | Classification metrics, vectorization, and classical baselines |
| **`pandas`** | **3.1.1** | Dataset manipulation and tabular results |
| **`matplotlib`** | **3.11.2** | High-resolution publication figure generation |
| **`seaborn`** | **0.13.2** | Statistical data visualization |
| **`datasets`** | **5.1.0** | Hugging Face benchmark corpus loader |

---

## 3. Activation Instructions

To activate this dedicated Qiskit environment:
```bash
# From the project root:
cd "/Users/bharatrathore/Desktop/expand/Bharat Data/padhai/B.Tech/sem7/NV62010 quantum computing and AI/Primer_QNN_Text_Classification-main"
source .venv-qiskit/bin/activate
```

