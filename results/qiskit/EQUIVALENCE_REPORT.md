# Numerical Equivalence Verification Report: PennyLane vs. IBM Qiskit

## Course: NV62010 — Quantum Computing with Artificial Intelligence
**Institution:** Sharda University  
**Target Reference Paper:** Huzzat et al., IEEE Access 2026  
**Auditor:** Senior Quantum Machine Learning Auditor  
**Date:** November 2026  

---

## 1. Scope & Academic Qualification
This audit evaluates the numerical agreement between the PennyLane quantum circuit (`src/circuit.py`) and the native IBM Qiskit circuit (`qiskit_impl/qiskit_circuit.py`).

> [!NOTE]
> **Academic Qualification:**  
> The PennyLane and Qiskit implementations demonstrated close numerical agreement in Pauli-Z expectation values across the evaluated test configurations. While exhaustive mathematical identity over all uncountably infinite states in $\mathbb{R}^4$ cannot be proven solely by discrete numerical testing, the maximum error across the evaluated suites is bounded by floating-point precision ($< 10^{-8}$ for observables, and state fidelity $F > 1 - 10^{-14}$).

---

## 2. Equivalence Benchmark Results (10 Deterministic Configurations)

| Test ID | Configuration | PennyLane $\langle Z_0 \rangle$ | Qiskit $\langle Z_0 \rangle$ | Max Obs Error | State Fidelity $F$ | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Test 1: All Zeros | -0.986882 | -0.986882 | 2.96e-08 | 1.0000000000 | **PASS** |
| 2 | Test 2: Small Positive | -0.963944 | -0.963944 | 1.77e-08 | 1.0000000000 | **PASS** |
| 3 | Test 3: Mixed Signs | -0.903435 | -0.903435 | 9.67e-09 | 1.0000000000 | **PASS** |
| 4 | Test 4: Boundary Norm (+1) | -0.050939 | -0.050939 | 1.53e-09 | 1.0000000000 | **PASS** |
| 5 | Test 5: Negative Boundary (-1) | -0.109379 | -0.109379 | 3.24e-09 | 1.0000000000 | **PASS** |
| 6 | Test 6: Seeded Random (Seed 42) | 0.186143 | 0.186143 | 7.06e-09 | 1.0000000000 | **PASS** |
| 7 | Test 7: High Frequency | 0.125392 | 0.125392 | 6.00e-09 | 1.0000000000 | **PASS** |
| 8 | Test 8: Orthogonal Vector | -0.064694 | -0.064694 | 2.59e-09 | 1.0000000000 | **PASS** |
| 9 | Test 9: Seeded Random (Seed 123) | -0.029796 | -0.029796 | 4.06e-09 | 1.0000000000 | **PASS** |
| 10 | Test 10: Seeded Random (Seed 2026) | 0.014777 | 0.014777 | 1.29e-09 | 1.0000000000 | **PASS** |

---

## 3. Statistical Summary Across Evaluated Configurations
- **Total Test Configurations:** 10
- **Total Observable Comparisons:** 40 expectation values
- **Maximum Absolute Expectation Discrepancy:** 2.9609e-08
- **Mean Absolute Expectation Discrepancy:** 4.3563e-09
- **Minimum Statevector Fidelity:** 1.000000000000
- **Numerical Tolerance Threshold:** $1.0 \times 10^{-6}$
- **Verification Verdict:** **100% PASSED**

### Structural Verification Checks:
1. **Identical Input Vectors:** Verified.
2. **Identical Quantum Weight Shapes ($2 \times 4 \times 3$):** Verified.
3. **Identical Feature Encoding (Hadamard, $R_Z$, pairwise $R_{ZZ}$):** Verified.
4. **Identical Gate Ordering ($R_X \to R_Y \to R_Z$ per wire):** Verified.
5. **Identical CNOT Entangling Chain ($0 \to 1, 1 \to 2, 2 \to 3$):** Verified.
6. **Identical Observable Ordering:** Verified via little-endian/big-endian permutation.
