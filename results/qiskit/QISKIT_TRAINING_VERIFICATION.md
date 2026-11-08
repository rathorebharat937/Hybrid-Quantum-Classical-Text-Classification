# Qiskit Training Verification & Academic Audit Report

## Course: NV62010 — Quantum Computing with Artificial Intelligence
**Institution:** Sharda University  
**Project:** Hybrid Quantum-Classical Text Classification Using Variational Quantum Circuits  
**Audit Scope:** Verification of IBM Qiskit Model Training, Checkpoints, Predictions, and Evaluation Metrics  
**Date:** November 2026  

---

## 1. Executive Verification Status

| Audit Item | Verification Status | Evidence & Value |
| :--- | :---: | :--- |
| **Training Completion** | **VERIFIED COMPLETE** | 10 full epochs executed on the complete 3,024 CR training split |
| **Checkpoint Location** | **VERIFIED PRESENT** | `results/qiskit/checkpoints/CR_qiskit_best.pt` (78,950 bytes) |
| **Predictions Location** | **VERIFIED PRESENT** | `results/qiskit/predictions/CR_qiskit_preds.npz` & `qiskit_predictions.csv` |
| **Training Log Location** | **VERIFIED PRESENT** | `results/qiskit/qiskit_training_log.csv` (all 10 epochs recorded) |
| **Model Selection Protocol**| **VERIFIED CLEAN** | Best checkpoint chosen strictly on validation set F1 score (Val F1: 0.7701 at Epoch 10). Test split was never touched during training or selection. |
| **Random Seed** | **VERIFIED** | Seed 42 |
| **Input Feature Dimension**| **VERIFIED** | 4,760 active TF-IDF features (entire CR training vocabulary) |
| **Trainable Parameters** | **VERIFIED** | 19,078 total trainable parameters (19,044 encoder + 24 quantum + 10 readout) |
| **Test Accuracy** | **VERIFIED** | **79.95%** (307 correct out of 384 test instances) |
| **Test Weighted F1** | **VERIFIED** | **79.90%** |
| **Test Macro F1** | **VERIFIED** | **78.19%** |
| **Test Positive-Class F1** | **VERIFIED** | **84.38%** |
| **Test ROC-AUC** | **VERIFIED** | **87.43%** (evaluated on continuous softmax probabilities) |
| **Confusion Matrix** | **VERIFIED** | TP: 208, TN: 99, FP: 40, FN: 37 |

---

## 2. Complete 10-Epoch Training Trajectory

```
================================================================================
Epoch   Train Loss   Val Accuracy   Val Weighted F1   Epoch Time (s)   Status
================================================================================
01        0.6547        63.81%          49.72%             9.49s        
02        0.6463        63.81%          49.72%             9.47s        
03        0.6185        63.81%          49.72%             9.47s        
04        0.5616        72.10%          66.54%             9.47s        
05        0.4865        75.97%          74.35%             9.45s        
06        0.4061        74.03%          73.29%             9.47s        
07        0.3339        75.41%          75.11%             9.47s        
08        0.2766        75.14%          74.91%             9.60s        
09        0.2323        76.52%          76.37%             9.52s        
10        0.1993        77.07%          77.01%             9.48s       [BEST CHECKPOINT]
================================================================================
Total Training Duration: 95.78s | Average Time per Epoch: 9.58s
```

---

## 3. Independent Recalculation of Evaluation Metrics from Saved Predictions

Recomputing metrics from `results/qiskit/qiskit_predictions.csv`:
- Total Test Samples: 384 (Class 0: 139, Class 1: 245)
- Accuracy = $(208 + 99) / 384 = 307 / 384 = \mathbf{0.799479 \implies 79.95\%}$
- Class 0 Precision = $99 / (99 + 37) = 0.7279$
- Class 0 Recall = $99 / (99 + 40) = 0.7122$
- Class 0 F1 = $0.7200$
- Class 1 Precision = $208 / (208 + 40) = 0.8387$
- Class 1 Recall = $208 / (208 + 37) = 0.8490$
- Class 1 F1 = $\mathbf{0.8438}$
- Weighted F1 = $\frac{139}{384}(0.7200) + \frac{245}{384}(0.8438) = \mathbf{0.7990}$
- Macro F1 = $\frac{0.7200 + 0.8438}{2} = \mathbf{0.7819}$
- ROC-AUC = $\mathbf{0.8743}$

