# Parameter Count Audit & Verification Report

## Course: NV62010 — Quantum Computing with Artificial Intelligence
**Institution:** Sharda University  
**Project:** Design and Implementation of a Hybrid Quantum-Classical Neural Network for Binary Text Classification  
**Date of Audit:** November 2026  
**Auditor:** Senior Quantum Machine Learning & PyTorch Auditor

---

## 1. Resolution of the 20,038 vs. 19,078 Discrepancy

### The Problem:
Earlier documentation claimed that the Hybrid Quantum-Classical Neural Network contains **20,038 total parameters**, whereas instantiated Qiskit models and logs reported **19,078 parameters**.

### The Investigation & Mathematical Proof:
The discrepancy arises from the difference between the **configured upper bound** (`max_features=5000`) and the **actual vocabulary size** extracted by `TfidfVectorizer` from the training split:

1. **Movie Reviews (MR) & Stanford Sentiment Treebank (SST):**
   - The training vocabulary exceeds 5,000 unique terms.
   - `TfidfVectorizer(max_features=5000)` extracts exactly 5,000 features.
   - **Theoretical & MR/SST Parameter Calculation:**
     - Classical Encoder (`fc1`): `Linear(5000, 4)` $\implies 5000 \times 4 \text{ weights} + 4 \text{ biases} = 20,004$
     - Quantum Variational Layer (`qlayer`): $2 \text{ layers} \times 4 \text{ qubits} \times 3 \text{ angles} = 24$
     - Classical Readout Layer (`fc2`): `Linear(4, 2)` $\implies 4 \times 2 \text{ weights} + 2 \text{ biases} = 10$
     - **Total Trainable Parameters:** $20,004 + 24 + 10 = \mathbf{20,038}$

2. **Customer Reviews (CR):**
   - The entire CR training partition contains only **4,760 unique vocabulary tokens**.
   - Consequently, `TfidfVectorizer(max_features=5000)` caps at **4,760 active features**.
   - **Actual CR Instantiated Parameter Calculation:**
     - Classical Encoder (`fc1`): `Linear(4760, 4)` $\implies 4760 \times 4 \text{ weights} + 4 \text{ biases} = 19,044$
     - Quantum Variational Layer (`qlayer`): $2 \text{ layers} \times 4 \text{ qubits} \times 3 \text{ angles} = 24$
     - Classical Readout Layer (`fc2`): `Linear(4, 2)` $\implies 4 \times 2 \text{ weights} + 2 \text{ biases} = 10$
     - **Total Trainable Parameters:** $19,044 + 24 + 10 = \mathbf{19,078}$

### Conclusion:
Both figures are correct within their respective contexts, but must be explicitly labeled:
- **Theoretical Architecture / 5000-feature datasets (MR, SST):** **20,038 parameters**
- **Actual CR Model Instance (4,760 active features):** **19,078 parameters**

---

## 2. Complete Model Parameter Audit (Customer Reviews Dataset)

| Model Name | Framework | Architecture / Layer Dimensions | Trainable Parameters | Fitted Coefficients (Sklearn) |
| :--- | :--- | :--- | :---: | :---: |
| **Logistic Regression** | Scikit-Learn | Linear model with L2 regularization | 0 (PyTorch) | 4,761 (4760 weights + 1 intercept) |
| **Linear SVM** | Scikit-Learn | Linear support vector classifier | 0 (PyTorch) | 4,761 (4760 weights + 1 intercept) |
| **Classical Deep MLP** | PyTorch | `Linear(4760, 64) -> ReLU -> Linear(64, 32) -> ReLU -> Linear(32, 2)` | 306,850 | N/A |
| **Compact MLP (4-dim)** | PyTorch | `Linear(4760, 4) -> Tanh -> Linear(4, 2)` | 19,054 | N/A |
| **PennyLane Hybrid QNN** | PennyLane + PyTorch | `Linear(4760, 4) -> Tanh -> VQC(4q, 2L) -> Linear(4, 2)` | 19,078 | N/A |
| **IBM Qiskit Hybrid QNN** | IBM Qiskit + PyTorch | `Linear(4760, 4) -> Tanh -> VQC(4q, 2L) -> Linear(4, 2)` | 19,078 | N/A |

### Parameter Breakdown for Instantiated Qiskit Model:
```
Module: QiskitHybridClassifier
--------------------------------------------------------------------------------
Layer Name           Shape                  Element Count       Data Type
--------------------------------------------------------------------------------
fc1.weight           torch.Size([4, 4760])  19,040              torch.float32
fc1.bias             torch.Size([4])        4                   torch.float32
qlayer.weights       torch.Size([2, 4, 3])  24                  torch.float32
fc2.weight           torch.Size([2, 4])     8                   torch.float32
fc2.bias             torch.Size([2])        2                   torch.float32
--------------------------------------------------------------------------------
Total Trainable Parameters:                 19,078
```

