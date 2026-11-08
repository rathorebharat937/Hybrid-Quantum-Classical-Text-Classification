# Runtime & Computational Complexity Audit Report

## Course: NV62010 — Quantum Computing with Artificial Intelligence
**Institution:** Sharda University  
**Auditor:** Senior Quantum Machine Learning Auditor  
**Dataset:** Customer Reviews (CR) — Train: 3,024, Val: 362, Test: 384  
**Date:** November 2026  

---

## 1. Fair, Compatible Runtime Comparison Table

| Model Architecture | Framework | Total Training Time | Average Time / Epoch | Test Eval Time | Inference Latency | Simulator / Backend | Gradient Method |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Logistic Regression** | Scikit-Learn | 0.123s | N/A (Convex) | 0.0008s | 0.002 ms/sample | Native CPU | L-BFGS |
| **Linear SVM** | Scikit-Learn | 0.150s | N/A (Convex) | 0.0041s | 0.011 ms/sample | Native CPU | Coordinate Descent |
| **Compact MLP (4-dim)** | PyTorch | 0.288s | 0.029s/ep | 0.0017s | 0.005 ms/sample | PyTorch Tensor | Reverse-Mode AD |
| **Hybrid QNN (PennyLane)** | PennyLane | 5.219s | 0.522s/ep | 0.0057s | 0.015 ms/sample | `default.qubit` | Reverse-Mode AD (tensor graph) |
| **Hybrid QNN (IBM Qiskit)** | IBM Qiskit | 1341.8s (10 ep) | 134.18s/ep | 0.3076s | 0.801 ms/sample | Statevector | Parameter-Shift (weights) + FD (inputs) |

> [!IMPORTANT]
> **Clarification of Timing Units:**  
> Previous versions of documentation incorrectly placed the total 10-epoch runtime of PennyLane (6.06s) alongside the per-epoch runtime of Qiskit (~132.8s/ep).  
> In matched 10-epoch comparisons:
> - **PennyLane QNN:** Trains at **0.522 seconds per epoch** (Total: 5.22s).
> - **IBM Qiskit QNN:** Trains at **134.18 seconds per epoch** (Total 10 epochs: 1341.8s).

---

## 2. Rigorous Quantum Circuit Execution Count Derivation

The previous claim of **172,368 circuit executions per epoch** for IBM Qiskit is mathematically verified as follows:

For each sample $b \in \{1, \dots, 3024\}$:
1. **Forward Pass:** 1 full circuit evaluation.
2. **Backward Pass — Variational Weights:**  
   $2 \text{ layers} \times 4 \text{ qubits} \times 3 \text{ rotation parameters} = 24 \text{ parameters}$.  
   Under the two-point parameter-shift rule ($s = \pi/2$), each parameter requires 2 shifted circuit evaluations:  
   $24 \times 2 = \mathbf{48} \text{ circuit evaluations}$.
3. **Backward Pass — Classical Input Encoder Gradients:**  
   4 continuous bottleneck features $x_0, x_1, x_2, x_3$.  
   Under symmetric central finite difference, each feature requires 2 shifted circuit evaluations:  
   $4 \times 2 = \mathbf{8} \text{ circuit evaluations}$.

$$\text{Total Logical Circuits per Sample} = 1 + 48 + 8 = 57 \text{ circuits}$$

$$\text{Total Logical Circuits per Epoch} = 3,024 \text{ samples} \times 57 = \mathbf{172,368 \text{ circuit evaluations per epoch}}$$

$$\text{Total for 10 Epochs} = 172,368 \times 10 = \mathbf{1,723,680 \text{ circuit evaluations}}$$

### Why PennyLane is Faster on CPU Simulators:
PennyLane's `default.qubit` implements quantum operations directly using PyTorch complex tensor operations. Consequently, PyTorch's C++ autograd engine differentiates the entire statevector graph via reverse-mode automatic differentiation (adjoint method) without constructing or executing individual parameter-shifted circuits.  
On physical quantum hardware (where statevectors cannot be observed directly), parameter-shift is required, making Qiskit's circuit count realistic for physical quantum processors.
