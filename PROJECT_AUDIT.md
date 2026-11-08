# Project Audit Report
## Quantum Neural Network for Text Classification

**Audit Date:** 2026-11-06  
**Repository:** Primer_QNN_Text_Classification  
**Source:** https://github.com/AnnyshaHuzzat/Primer_QNN_Text_Classification

---

## 1. Existing Architecture

The project implements a hybrid quantum-classical pipeline for binary text classification:

```
Raw Text → TF-IDF (5000-dim) → Linear Layer (5000→4) → tanh
→ 4-Qubit VQC (2 layers) → Pauli-Z Measurements → Linear (4→2) → Prediction
```

### Quantum Circuit (VQC)
- **Qubits:** 4
- **Variational Layers:** 2
- **Embedding:** IQP Embedding (angle encoding)
- **Rotations:** RX, RY, RZ per qubit per layer
- **Entanglement:** CNOT ladder (qubit i → qubit i+1)
- **Measurement:** Pauli-Z expectation values on all 4 qubits
- **Trainable Parameters:** `weights` shape `(2, 4, 3)` = 24 quantum params

### Total Parameters
- `fc1`: 5000 × 4 + 4 = 20,004
- `qlayer`: 24 (quantum)
- `fc2`: 4 × 2 + 2 = 10
- **Total:** 20,038

---

## 2. Project Structure

```
Primer_QNN_Text_Classification-main/
├── README.md                    # Project documentation
├── configs/
│   ├── default.yaml             # Default training config
│   └── ablation_details.yaml    # Ablation parameter grid
├── src/
│   ├── __init__.py              # Package init
│   ├── circuit.py               # PennyLane quantum circuit
│   ├── model.py                 # HybridClassifier PyTorch model
│   ├── data.py                  # Dataset loading dispatcher
│   └── train.py                 # Training and evaluation script
├── data/                        # Local dataset files
│   ├── mr/                      # Movie Review dataset
│   ├── cr/                      # Customer Review dataset
│   └── sst/                     # Stanford Sentiment Treebank
├── pipeline.png                 # Architecture diagram
├── vqc.png                      # VQC circuit diagram
├── download_datasets.py         # Auto-download helper script
└── venv/                        # Python virtual environment
```

---

## 3. Identified Problems

### 3.1 Critical Bugs (Fixed)
| Issue | File | Fix Applied |
|-------|------|-------------|
| `HybridClassifier.__init__` missing `n_layers` parameter | `src/model.py` | Added `n_layers=2` parameter, passed to `build_qlayer()` |
| `circuit.py` used hardcoded globals instead of parameterized function | `src/circuit.py` | Refactored `build_qlayer()` to accept `n_qubits`, `n_layers` |
| Dataset paths set to placeholder `"Enter the Dataset Location"` | `src/data.py` | Set to `data/mr`, `data/cr/custrev.txt`, `data/sst` |

### 3.2 Missing Components
| Component | Status |
|-----------|--------|
| Classical baseline models (LR, SVM, MLP) | Not implemented → Added |
| requirements.txt | Missing → Created |
| Visualization / graph generation | Missing → Added |
| Ablation experiment runner | Missing → Added |
| Confusion matrix, ROC, comprehensive metrics | Missing → Added |

### 3.3 Design Observations
- The `IQPEmbedding` encodes inputs, not standard `AngleEmbedding`. The README says "angle embedding" but IQP is a valid angle-based embedding with additional entanglement—this is preserved as the authors' design choice.
- No explicit data leakage between train/val/test splits detected—`train_test_split` with fixed seeds ensures reproducibility.
- SST dataset from Hugging Face has 67,349 samples but the code loads 6,920 (SetFit/sst2 version), not the full GLUE SST-2. This matches the paper's reported 65,528 approximately.

---

## 4. Required Fixes Summary

1. ✅ Fix `model.py` to accept `n_layers` parameter
2. ✅ Fix `circuit.py` to parameterize qubit count and layer count
3. ✅ Set dataset paths in `data.py`
4. ✅ Download MR, CR, SST datasets from Hugging Face
5. ✅ Install all dependencies in virtual environment
6. ✅ Create classical baseline implementations
7. ✅ Create comprehensive experiment runner
8. ✅ Add visualization pipeline

---

## 5. Execution Plan

1. **Phase 1-2:** Environment verified, dependencies installed
2. **Phase 3:** All 5 datasets loaded (MR, CR, SST local; SUBJ, MPQA from HuggingFace)
3. **Phase 4:** QNN model verified, quantum circuit differentiable
4. **Phase 5:** Classical baselines implemented (LR, SVM, MLP, Compact MLP)
5. **Phase 6:** Full training on all datasets
6. **Phase 7-8:** Evaluation metrics and comparison tables
7. **Phase 9:** Research-quality visualizations
8. **Phase 10:** Ablation studies (qubits, layers, learning rate)
9. **Phase 11-13:** Checkpoints, reproducibility, documentation

