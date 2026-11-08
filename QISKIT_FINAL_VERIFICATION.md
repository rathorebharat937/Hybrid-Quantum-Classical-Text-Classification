# Final Qiskit Scientific Verification and Academic Audit Report

## Course: NV62010 — Quantum Computing with Artificial Intelligence
**Academic Institution:** Sharda University  
**Degree & Semester:** Bachelor of Technology (B.Tech), Semester 7  
**Project Title:** Design and Implementation of a Hybrid Quantum-Classical Neural Network for Binary Text Classification Using Variational Quantum Circuits  
**Base Reference Paper:** *"A Primer on Quantum Neural Network Model for Text Classification: From Concept to Design"*, IEEE Access, 2026 (Huzzat et al.)  
**Lead Auditor:** Senior Quantum Computing Engineer & PyTorch ML Auditor  
**Date of Audit Completion:** November 2026  

---

## 1. Master Scientific Verification Table

| Claim / Item | Evidence Source | Verification Method | Status | Final Correct Verified Value |
| :--- | :--- | :--- | :---: | :--- |
| **Qiskit Circuit Execution** | `results/qiskit/circuit_execution.json` & `circuit_diagram.txt` | Executed forward simulation on statevector backend | **PASS** | Successfully executes forward simulation; outputs statevectors and Pauli-$Z$ expectations. |
| **Circuit has 44 Operations** | `qiskit_impl/qiskit_circuit.py` & `circuit_execution.json` | Exact gate counting via Qiskit DAG inspection | **PASS** | **44 total quantum gates** (4 H, 4 $R_Z$, 6 $R_{ZZ}$, 24 variational single-qubit rotations, 6 CNOTs), Depth = 17. |
| **PennyLane-Qiskit Equivalence**| `results/qiskit/equivalence_results.csv` & `EQUIVALENCE_REPORT.md` | Recomputed across 10 deterministic suites + state fidelity | **PASS** | Max observable error: **$2.96 \times 10^{-8}$**, Minimum Statevector Fidelity: **$F > 1.0 - 10^{-14}$**. |
| **Gradients are Correct** | `results/qiskit/gradient_validation.csv` & `GRADIENT_VALIDATION_REPORT.md` | Numerical finite difference vs parameter-shift ($s=\pi/2$) | **PASS** | Weight gradient error: $\mathbf{< 5.1 \times 10^{-9}}$, Input feature gradient error: $\mathbf{< 2.5 \times 10^{-6}}$. |
| **Qiskit Training Completed** | `results/qiskit/qiskit_training_log.csv` & `CR_qiskit_best.pt` | Verified all 10 epochs executed on complete CR train set | **PASS** | **10 complete epochs** executed on all 3,024 CR instances; training completed in 95.78s. |
| **Qiskit Test Accuracy** | `results/qiskit/predictions/CR_qiskit_preds.npz` & `qiskit_predictions.csv` | Recomputed metrics directly from saved test predictions | **PASS** | **Accuracy: 79.95%** (307/384), Weighted F1: **79.90%**, Macro F1: **78.19%**, AUC: **87.43%**. *(Previous 79.69% was unverified).* |
| **Model Parameter Count** | `results/qiskit/parameter_count_audit.csv` & `PARAMETER_COUNT_VERIFICATION.md` | Programmatic counting of instantiated PyTorch tensors | **PASS** | **19,078 parameters** on CR (due to 4,760 vocabulary features) vs **20,038 parameters** on theoretical 5,000-feature datasets. |
| **Runtime Comparison is Fair**| `results/qiskit/runtime_audit.csv` & `RUNTIME_COMPARISON_REPORT.md` | Separated total training from per-epoch duration | **PASS** | Replaced misleading 0.33s vs 132.8s/ep comparison with matched 10-epoch benchmarks. Derived 172,368 circuits/epoch. |
| **Report Tables are Consistent**| `results/qiskit/three_way_comparison.csv` & `qiskit_model_results.csv` | Cross-checked CSVs across all report files | **PASS** | All markdown tables in `FINAL_PROJECT_REPORT.md` and `QISKIT_IMPLEMENTATION_REPORT.md` match exact CSV values. |

---

## 2. Detailed Audit Findings & Corrections Implemented

### Issue 1: Missing / Unverified Qiskit Training Results
- **Previous State:** Benchmark script printed `Warning: results/qiskit/qiskit_model_results.csv not found yet.` Earlier documentation claimed 79.69% accuracy based on placeholder/unverified projections.
- **Root Cause:** The original training process was running in the background and was interrupted before finishing.
- **Correction Applied:** Executed the complete 10-epoch training on the entire Customer Reviews (CR) dataset (3,024 training instances, batch size 32, Adam optimizer, lr=0.001, seed 42) using the optimized batched statevector implementation.
- **Verified Empirical Result:** The genuine trained model achieved **79.95% accuracy**, **79.90% weighted F1**, **78.19% macro F1**, **84.38% positive-class F1**, and **87.43% ROC-AUC**. Checkpoint saved to `results/qiskit/checkpoints/CR_qiskit_best.pt`, predictions saved to `results/qiskit/qiskit_predictions.csv`, and training trajectory saved to `results/qiskit/qiskit_training_log.csv`.

### Issue 2: Parameter Count Discrepancy (20,038 vs 19,078)
- **Previous State:** Project documentation alternated between 20,038 and 19,078 parameters without explanation.
- **Root Cause:** On datasets with large vocabularies (MR, SST), `TfidfVectorizer(max_features=5000)` yields 5,000 features $\implies 5000 \times 4 + 4 + 24 + 8 + 2 = 20,038$. However, the CR training corpus has only 4,760 unique tokens, capping active features at 4,760 $\implies 4760 \times 4 + 4 + 24 + 8 + 2 = 19,078$.
- **Correction Applied:** Documented the distinction in `results/qiskit/PARAMETER_COUNT_VERIFICATION.md` and `parameter_count_audit.csv`. All reports now clearly label theoretical 5000-feature vs instantiated CR model parameter counts.

### Issue 3: Misleading Runtime Comparisons
- **Previous State:** Reports juxtaposed PennyLane total training time (6.06s for 10 epochs) with Qiskit per-epoch time (~132.8s/epoch).
- **Correction Applied:** Created `results/qiskit/RUNTIME_COMPARISON_REPORT.md` and `runtime_audit.csv`, explicitly separating setup time, TF-IDF time, total training time, per-epoch time, evaluation time, and inference latency per sample.
- **Circuit Evaluation Count:** Mathematically verified the exact logical circuit count: 57 circuit evaluations per sample ($1 \text{ fwd} + 48 \text{ weight shifts} + 8 \text{ input shifts}$) $\times 3,024 \text{ samples} = \mathbf{172,368 \text{ circuit evaluations per epoch}}$ ($1,723,680$ evaluations over 10 epochs).

### Issue 4: Equivalence Claims and Gradient Validation
- **Previous State:** Equivalence claims were based solely on 8 Pauli-$Z$ observable tests.
- **Correction Applied:** Expanded the suite to 10 deterministic test cases, added full 16-element quantum statevector fidelity checks ($F > 1 - 10^{-14}$), and validated analytical parameter-shift gradients against high-precision central finite difference ($< 5.1 \times 10^{-9}$ error).

---

## 3. Outstanding Issues Before University Submission

**Status:** **ALL SCIENTIFIC, CODE, AND REPORT INCONSISTENCIES ARE FULLY RESOLVED.**

Every claim in the project reports is backed by verified source code, executed scripts, saved PyTorch checkpoints, and genuine CSV datasets:
- No placeholder or fabricated numbers remain.
- All 10 epochs of Qiskit training on the CR dataset are completed and verified.
- The 20,038 vs 19,078 parameter discrepancy is mathematically proven.
- Statevector simulation vs physical hardware execution is transparently disclosed.
- The lack of demonstrated quantum advantage over the classical Compact MLP baseline is honestly documented.

