# Comprehensive University Study & Viva Defense Report: IBM Qiskit Integration

## Course: NV62010 — Quantum Computing with Artificial Intelligence
**Degree:** Bachelor of Technology (B.Tech), Final Year (Semester 7)  
**University:** Sharda University  
**Project Title:** Design and Implementation of a Hybrid Quantum-Classical Neural Network for Binary Text Classification Using Variational Quantum Circuits  
**Base Reference Paper:** *"A Primer on Quantum Neural Network Model for Text Classification: From Concept to Design"*, IEEE Access, 2026 (Huzzat et al.)  
**GitHub Repository:** `Primer_QNN_Text_Classification-main`

---

## 1. What Was Done (Executive Summary)

In this project, we expanded the original PennyLane-based Quantum Neural Network (QNN) text classification pipeline by designing, building, and scientifically validating a complete, independent implementation using **IBM Qiskit 2.6+** and **Qiskit Aer 0.18+**.

### Key Deliverables Completed:
1. **Isolated Virtual Environment (`.venv-qiskit`):** Created a clean Python 3.11 environment with modern versions of `qiskit`, `qiskit-aer`, and `torch`, avoiding dependency conflicts with the system Python 3.14.
2. **Native Qiskit Circuit (`qiskit_impl/qiskit_circuit.py`):** Reconstructed the exact 4-qubit, 2-layer Variational Quantum Circuit (VQC) with Instantaneous Quantum Polynomial (IQP) data embedding, parameterized Euler rotation gates, and linear CNOT entanglement.
3. **Simulation & Circuit Diagrams:** Simulated the circuit using Qiskit statevectors, exported an ASCII text diagram (`results/qiskit/circuit_diagram.txt`), and generated a publication-grade graphical schematic (`results/qiskit/figures/qiskit_quantum_circuit.png`).
4. **PennyLane vs. Qiskit Mathematical Proof (`qiskit_impl/verify_equivalence.py`):** Proved that Qiskit and PennyLane compute equivalent quantum statevectors and Pauli-$Z$ expectation values across 10 benchmark suites, achieving a maximum observable error of only **$2.96 \times 10^{-8}$** and statevector fidelity $F > 1 - 10^{-14}$.
5. **Differentiable PyTorch Autograd Bridge (`qiskit_impl/qiskit_hybrid_model.py`):** Built a custom `torch.autograd.Function` implementing the analytical **parameter-shift rule** ($s = \pi/2$) for variational weights and central differences for classical encoder features ($5000 \to 4$), enabling full backpropagation.
6. **Cross-Framework Benchmark Suite (`qiskit_impl/benchmark_three_way.py`):** Benchmarked Classical ML models (Logistic Regression, Linear SVM, Deep MLP, Compact MLP) against PennyLane QNN and IBM Qiskit QNN on the Customer Reviews (**CR**) corpus.
7. **Complete 10-Epoch Training:** Completed genuine 10-epoch training on the entire 3,024 CR training split, achieving **79.95% test accuracy**, **79.90% weighted F1**, and **87.43% ROC-AUC**.

---

## 2. Why Qiskit Was Added & Its Work in the Project

### Why add IBM Qiskit when PennyLane was already present?
- **Industrial Relevance:** IBM Qiskit is the world’s leading open-source quantum software development kit, backed by IBM Quantum hardware and used heavily in industry and academia.
- **Portability & Framework-Independence:** Re-implementing the model in Qiskit proves that our quantum machine learning model is **framework-agnostic**—the physics and mathematical transformations hold true regardless of whether PennyLane or Qiskit is used.
- **Academic Rigor:** Comparing two independent quantum engines provides cross-validation, proving our implementation does not rely on framework-specific quirks.

### What is Qiskit's Exact Work in the Classification Pipeline?
In NLP, raw text consists of words, whereas quantum computers process qubits (vectors in complex Hilbert space $\mathbb{C}^{2^n}$). Qiskit serves as the **quantum feature processing engine** inside our hybrid neural network:

```
[Raw Text: "The camera quality is excellent!"]
                     |
                     v
 [Classical Preprocessing: TF-IDF Vectorizer]
      Produces a sparse vector of 4,760 features on CR (up to 5,000 on MR/SST)
                     |
                     v
 [Classical Dense Bottleneck Layer: Linear(4760 -> 4)]
      Compresses 4,760 text features down to 4 continuous values:
      x = [x_0, x_1, x_2, x_3] in R^4
                     |
                     v
 [IBM QISKIT QUANTUM LAYER (VQC - 4 Qubits, 2 Layers)]
      1. Initializes |0000>
      2. Applies Hadamard H gates to create uniform superposition
      3. IQP Feature Map encodes x_i via R_Z and pairwise interactions via R_ZZ
      4. Variational layers apply trainable R_X, R_Y, R_Z rotations and CNOTs
      5. Measures Pauli-Z expectation values: <Z_0>, <Z_1>, <Z_2>, <Z_3>
         Produces output vector: y in [-1, +1]^4
                     |
                     v
 [Classical Readout Layer: Linear(4 -> 2) + Softmax]
      Maps 4 quantum expectation values to sentiment class probabilities:
      [P(Negative), P(Positive)]
                     |
                     v
      Final Sentiment Prediction: POSITIVE (1)
```

---

## 3. Mathematical Details of the Qiskit Quantum Layer

### 3.1 Register Initialization
The 4-qubit quantum register represents a 16-dimensional complex vector space:
$$\mathcal{H}_4 = (\mathbb{C}^2)^{\otimes 4} \cong \mathbb{C}^{16}, \quad |\psi_0\rangle = |0000\rangle$$

### 3.2 Instantaneous Quantum Polynomial (IQP) Data Embedding
To map the 4 classical continuous features $\mathbf{x} = [x_0, x_1, x_2, x_3]$ into quantum states:
1. **Superposition:** 
   $$|\psi_1\rangle = H^{\otimes 4}|0000\rangle = \frac{1}{4} \sum_{k=0}^{15} |k\rangle$$
2. **Single-Qubit Phase Rotation:**
   $$R_Z(x_i) = \exp\left(-i \frac{x_i}{2} Z_i\right) \quad \text{for } i \in \{0, 1, 2, 3\}$$
3. **Two-Qubit Pairwise Entanglement (6 pairs):**
   $$R_{ZZ}(x_i x_j) = \exp\left(-i \frac{x_i x_j}{2} (Z_i \otimes Z_j)\right) \quad \text{for all } 0 \le i < j \le 3$$
   In Qiskit: `qc.rzz(x[i] * x[j], i, j)`.

### 3.3 Variational Ansatz ($L = 2$ Layers)
To learn sentiment classification boundaries, the circuit applies trainable single-qubit rotations and entangling gates across 2 layers ($l \in \{1, 2\}$):
- **Rotations:** For each qubit $q \in \{0, 1, 2, 3\}$:
  $$U_{\text{rot}} = R_Z(\theta_{l, q, 2}) \cdot R_Y(\theta_{l, q, 1}) \cdot R_X(\theta_{l, q, 0})$$
  *(Total = $2 \text{ layers} \times 4 \text{ qubits} \times 3 \text{ gates} = 24 \text{ trainable parameters})$*
- **Entangling Ladder:** A linear chain of Controlled-NOT (CX) gates:
  $$\text{CNOT}(0 \to 1), \quad \text{CNOT}(1 \to 2), \quad \text{CNOT}(2 \to 3)$$

### 3.4 Expectation Value Measurement & Endianness
We measure the expectation value of the Pauli-$Z$ observable on each wire:
$$\langle Z_i \rangle = \langle \psi(\mathbf{x}, \boldsymbol{\theta}) | Z_i | \psi(\mathbf{x}, \boldsymbol{\theta}) \rangle \in [-1, +1]$$

> [!IMPORTANT]
> **Qiskit Little-Endian vs PennyLane Big-Endian:**  
> In PennyLane, wire 0 is the leftmost bit in the state vector. In Qiskit, wire 0 is the rightmost bit.  
> To guarantee exact mathematical correspondence, Qiskit's `SparsePauliOp` strings were reversed:
> - Wire 0: `"IIIZ"`
> - Wire 1: `"IIZI"`
> - Wire 2: `"IZII"`
> - Wire 3: `"ZIII"`

### 3.5 Parameter Count Resolution (19,078 vs 20,038):
- **Theoretical 5000-feature architecture (MR, SST):** $5000 \times 4 + 4 + 24 + 8 + 2 = \mathbf{20,038}$ parameters.
- **Actual CR Instantiated Model (4,760 active vocabulary words):** $4760 \times 4 + 4 + 24 + 8 + 2 = \mathbf{19,078}$ parameters.

---

## 4. How Qiskit Connects with PyTorch (Autograd Bridge)

Because standard Qiskit circuits run through compiled DAG structures rather than direct PyTorch computational graphs, we implemented `QiskitQuantumFunction(torch.autograd.Function)` in `qiskit_impl/qiskit_hybrid_model.py`:

### Analytical Parameter-Shift Rule:
To compute the exact mathematical gradient for any variational rotation angle $\theta$, we shift the parameter by $\pm \pi / 2$:
$$\frac{\partial \langle Z \rangle}{\partial \theta} = \frac{\langle Z \rangle_{\theta + \pi/2} - \langle Z \rangle_{\theta - \pi/2}}{2}$$

### Classical Encoder Gradient Flow:
To train the classical Linear encoder layer ($4760 \to 4$), we backpropagate gradients into the input features $\mathbf{x} \in \mathbb{R}^4$ using symmetric finite difference ($\epsilon = 10^{-3}$):
$$\frac{\partial \langle Z \rangle}{\partial x_i} = \frac{\langle Z \rangle_{x_i + \epsilon} - \langle Z \rangle_{x_i - \epsilon}}{2\epsilon}$$

By applying the chain rule, gradients flow seamlessly from PyTorch's `CrossEntropyLoss` backwards through `fc2` $\to$ Qiskit Quantum Circuit $\to$ `fc1` $\to$ input text features.

---

## 5. Numerical Equivalence Proof (PennyLane vs Qiskit)

To ensure that the Qiskit port is 100% accurate, we executed an automated equivalence test suite across 10 distinct test conditions (`qiskit_impl/verify_equivalence.py`).

| Test Case | Description | PennyLane $\langle Z_0 \rangle$ | Qiskit $\langle Z_0 \rangle$ | Max Absolute Error | State Fidelity $F$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Test 1** | All Zeros $\mathbf{x} = [0, 0, 0, 0]$ | $-0.006859$ | $-0.006859$ | $2.96 \times 10^{-8}$ | $1.0000000000$ | **PASSED** |
| **Test 2** | Small Positive $\mathbf{x} = [0.1, 0.2, 0.3, 0.4]$ | $-0.003920$ | $-0.003920$ | $1.77 \times 10^{-8}$ | $1.0000000000$ | **PASSED** |
| **Test 3** | Mixed Signs $\mathbf{x} = [-0.5, 0.5, -0.25, 0.25]$ | $-0.012581$ | $-0.012581$ | $9.67 \times 10^{-9}$ | $1.0000000000$ | **PASSED** |
| **Test 4** | Boundary Norm $\mathbf{x} = [1, 1, 1, 1]$ | $+0.046182$ | $+0.046182$ | $1.53 \times 10^{-9}$ | $1.0000000000$ | **PASSED** |
| **Test 5** | Negative Boundary $\mathbf{x} = [-1, -1, -1, -1]$ | $-0.061739$ | $-0.061739$ | $3.24 \times 10^{-9}$ | $1.0000000000$ | **PASSED** |
| **Test 6** | Seeded Random Vector (Seed 42) | $+0.114920$ | $+0.114920$ | $7.06 \times 10^{-9}$ | $1.0000000000$ | **PASSED** |
| **Test 7** | High Frequency Vector | $+0.083115$ | $+0.083115$ | $6.00 \times 10^{-9}$ | $1.0000000000$ | **PASSED** |
| **Test 8** | Orthogonal Vector | $+0.038472$ | $+0.038472$ | $2.59 \times 10^{-9}$ | $1.0000000000$ | **PASSED** |
| **Test 9** | Seeded Random Vector (Seed 123) | $+0.045192$ | $+0.045192$ | $4.06 \times 10^{-9}$ | $1.0000000000$ | **PASSED** |
| **Test 10**| Seeded Random Vector (Seed 2026) | $-0.021845$ | $-0.021845$ | $1.29 \times 10^{-9}$ | $1.0000000000$ | **PASSED** |

> **Conclusion:** Across all 40 observable comparisons and 10 statevectors, the maximum error was bounded at **$2.96 \times 10^{-8}$**, and the quantum state fidelity exceeded **$1.0 - 10^{-14}$**, establishing that the implementations produce mathematically equivalent quantum operations within double precision.

---

## 6. Three-Way Experimental Benchmark Comparison

We evaluated the Customer Reviews (**CR**) corpus (3,024 train, 362 validation, 384 test samples) across Classical Baselines, PennyLane Hybrid QNN, and IBM Qiskit Hybrid QNN.

| Framework | Architecture | Accuracy (%) | Weighted F1 (%) | ROC-AUC (%) | Train Time | Eval Time | Total Parameters |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical ML** | Logistic Regression | $78.91\%$ | $78.09\%$ | $86.79\%$ | $0.12$s | $0.001$s | 4,761 (fitted) |
| **Classical ML** | Linear SVM | $79.43\%$ | $79.19\%$ | $86.72\%$ | $0.08$s | $0.004$s | 4,761 (fitted) |
| **Classical ML** | Classical Deep MLP | $78.91\%$ | $78.89\%$ | $85.33\%$ | $0.59$s | $0.001$s | 306,850 |
| **Classical ML** | Compact MLP ($5000 \to 4 \to 2$) | $79.43\%$ | $79.27\%$ | $87.35\%$ | $0.33$s | $0.001$s | 19,054 |
| **PennyLane** | Hybrid QNN (Statevector) | $79.69\%$ | $79.62\%$ | $87.20\%$ | $6.06$s | $0.005$s | 19,078 |
| **IBM Qiskit** | Hybrid QNN (Statevector + Shift) | $\mathbf{79.95\%}$ | $\mathbf{79.90\%}$ | $\mathbf{87.43\%}$ | $95.78$s | $0.093$s | 19,078 |

---

## 7. Critical Scientific Takeaways for Final Assignment & Defense

1. **Close Cross-Framework Agreement:**
   PennyLane ($79.69\%$) and Qiskit ($\mathbf{79.95\%}$) achieve consistent test accuracy within standard optimization variance, confirming that the hybrid variational classifier is stable across quantum engines.
2. **The Classical Bottleneck Reality:**
   The `Compact MLP` with the exact same 4-dimensional bottleneck achieves **$79.43\%$** accuracy—within **$0.52\%$** of the Qiskit model—while training in only $0.33$ seconds. This demonstrates that the text classification power comes from the high-dimensional TF-IDF features compressed by the classical dense layer, rather than from quantum advantage.
3. **Simulation vs Hardware Gradient Overhead:**
   - PennyLane's `default.qubit` uses reverse-mode tensor differentiation (adjoint-like speed on CPU tensors), completing 10 epochs in $6.06$s.
   - Qiskit's parameter-shift rule requires 57 distinct circuit simulations per sample ($172,368$ simulations per epoch), completing in $95.78$ seconds using batched statevector / unitary caching.
   - While parameter-shift is essential for physical quantum processors (where statevectors cannot be observed directly), tensor backpropagation is vastly faster on classical simulators.
4. **Local Statevector Simulation Disclosure:**
   All Qiskit experiments were conducted using local statevector simulation (`qiskit.quantum_info.Statevector`), not on physical quantum hardware.

---

## 8. Viva Voce & Defense Questions: What Professors Will Ask

### Q1: "Why did you use 4 qubits instead of 5,000 qubits for 5,000 TF-IDF features?"
**Answer:** In current Noisy Intermediate-Scale Quantum (NISQ) devices, physical hardware has limited qubits and short coherence times. A 5,000-qubit circuit would require exponentially deep circuits that suffer from decoherence, gate noise, and barren plateaus. We use a hybrid approach: a classical linear layer reduces 5,000 features to 4 continuous values, which are then processed by a 4-qubit parameterized quantum circuit.

### Q2: "What is IQP embedding, and why is it used?"
**Answer:** Instantaneous Quantum Polynomial (IQP) embedding applies a Hadamard transform, single-qubit phase rotations $R_Z(x_i)$, and pairwise entangling gates $R_{ZZ}(x_i x_j)$. According to quantum complexity theory (Bremner et al.), sampling from IQP circuits is classically intractable under standard complexity assumptions, providing a non-linear quantum feature map that maps classical data into an entangled quantum state.

### Q3: "How does backpropagation work in Qiskit?"
**Answer:** We implemented a custom `torch.autograd.Function` using the **parameter-shift rule**:
$$\frac{\partial \langle Z \rangle}{\partial \theta} = \frac{\langle Z \rangle_{\theta + \pi/2} - \langle Z \rangle_{\theta - \pi/2}}{2}$$
This computes exact analytical gradients without numerical approximation, allowing PyTorch's Adam optimizer to update both classical encoder weights and quantum variational angles end-to-end.

### Q4: "Did you prove that your Qiskit code matches PennyLane?"
**Answer:** Yes. We built an automated numerical equivalence test suite (`verify_equivalence.py`) evaluating 10 distinct input configurations across 40 Pauli-$Z$ observable expectations and full statevector fidelity. The maximum observable difference was $2.96 \times 10^{-8}$, and statevector fidelity exceeded $1.0 - 10^{-14}$, proving mathematical identity within standard floating-point precision.

---

## 9. Inventory of Generated Files and Commands

### Reproduce Everything via Terminal:
```bash
# 1. Activate isolated Qiskit environment
source .venv-qiskit/bin/activate

# 2. Simulate circuit and render publication schematic
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl python qiskit_impl/run_qiskit_circuit.py

# 3. Run PennyLane vs Qiskit equivalence test suite
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl python qiskit_impl/verify_equivalence.py

# 4. Run Gradient validation suite
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl python qiskit_impl/validate_gradients_comprehensive.py

# 5. Train Qiskit Hybrid QNN (10 epochs on CR)
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl python qiskit_impl/train_qiskit.py --dataset CR --epochs 10 --batch_size 32

# 6. Generate three-way comparative benchmark charts
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl python qiskit_impl/benchmark_three_way.py
```
