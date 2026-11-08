# Scientific Validation & Research Audit Report
## Hybrid Quantum-Classical Neural Network for Text Classification

**Auditor:** Independent ML & QML Research Auditor  
**Audit Target:** Local Implementation & Experimental Outputs of `Primer_QNN_Text_Classification`  
**Date:** November 2026  
**Status:** Completed & Rigorously Validated  

---

## 1. Executive Summary & Core Audit Conclusions

1. **Metric Reproducibility:** Exact match across all 25 benchmark runs (0 discrepancies). Ground-truth prediction arrays in `results/predictions/*.npz` confirm reported numbers in `results/model_comparison.csv`.
2. **MPQA Dataset Exclusion:** An in-depth investigation of `jxm/mpqa` confirmed that short phrase repetitions (e.g., `"approval"`, `"to blame..."`) cause 271 exact overlaps between train and test sets. Because no document/source IDs exist in the dataset schema, a leakage-free split cannot be reconstructed from the public HF release. Consequently, **MPQA is formally excluded from clean benchmark claims and retained strictly as a documented dataset limitation.**
3. **Statistical Significance on CR & SST:** Across 5 independent random seeds (`42, 123, 2026, 7, 99`), paired $t$-tests and 95% confidence intervals between Hybrid QNN (2-layer) and the dimension-matched Compact MLP reveal:
   - On **CR:** Mean difference is $-0.0047$ (95% CI: $[-0.0142, +0.0048]$, $p = 0.2441$).
   - On **SST:** Mean difference is $+0.0069$ (95% CI: $[-0.0070, +0.0208]$, $p = 0.2383$).
   - **Rigorous Conclusion:** Because the 95% confidence intervals cross zero and $p > 0.05$ in both cases, **there is no statistically significant quantum advantage over the compact classical baseline at the 95% confidence level across 5 runs.** Claims of "quantum superiority" are scientifically unsubstantiated.
4. **Computational Trade-Off:** The 4-qubit parameterized quantum simulator introduces an approximate **$20\times$ wall-clock training time overhead** compared to the classical bottleneck MLP ($5.68\text{s}$ vs $0.26\text{s}$ on CR; $13.00\text{s}$ vs $0.64\text{s}$ on SST) while achieving comparable accuracy.

---

## 2. Statistical Analysis & Paired Significance Testing (Tasks 2 & 5)

To evaluate the specific contribution of the variational quantum circuit (VQC) relative to the classical 4-dimensional compression bottleneck ($5000 \to 4$), paired differences ($d_i = \text{QNN}_i - \text{CompactMLP}_i$) were computed across identical train/val/test splits and seeds.

### 2.1 Customer Reviews (CR) — 5-Seed Paired Statistical Analysis
| Metric | QNN (2-layer) Mean $\pm$ Std | Compact MLP Mean $\pm$ Std | Paired Mean Diff ($\bar{d}$) | 95% CI of Difference | Paired $t$-stat | $p$-value | Cohen's $d$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | $0.7943 \pm 0.0061$ | $0.7990 \pm 0.0050$ | $-0.0047$ (SE: $0.0034$) | $[-0.0142, +0.0048]$ | $-1.3646$ | $0.2441$ | $-0.6103$ |
| **Weighted F1** | $0.7931 \pm 0.0069$ | $0.7978 \pm 0.0053$ | $-0.0047$ (SE: $0.0039$) | $[-0.0154, +0.0060]$ | $-1.2299$ | $0.2861$ | $-0.5500$ |
| **AUC-ROC** | $0.8727 \pm 0.0005$ | $0.8724 \pm 0.0006$ | $+0.0003$ (SE: $0.0003$) | $[-0.0005, +0.0011]$ | $+1.1191$ | $0.3258$ | $+0.5005$ |

*Interpretation:* On CR, the Compact classical model performs slightly better on average ($-0.47\%$), with zero statistical difference ($p = 0.244$). The classical compression layer ($5000 \to 4$) accounts for the model's predictive ability.

### 2.2 Stanford Sentiment Treebank (SST) — 5-Seed Paired Statistical Analysis
| Metric | QNN (2-layer) Mean $\pm$ Std | Compact MLP Mean $\pm$ Std | Paired Mean Diff ($\bar{d}$) | 95% CI of Difference | Paired $t$-stat | $p$-value | Cohen's $d$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | $0.8105 \pm 0.0032$ | $0.8036 \pm 0.0085$ | $+0.0069$ (SE: $0.0050$) | $[-0.0070, +0.0208]$ | $+1.3850$ | $0.2383$ | $+0.6194$ |
| **Weighted F1** | $0.8103 \pm 0.0033$ | $0.8033 \pm 0.0085$ | $+0.0070$ (SE: $0.0050$) | $[-0.0069, +0.0210]$ | $+1.3945$ | $0.2356$ | $+0.6237$ |
| **AUC-ROC** | $0.8936 \pm 0.0024$ | $0.8907 \pm 0.0056$ | $+0.0029$ (SE: $0.0031$) | $[-0.0056, +0.0114]$ | $+0.9357$ | $0.4024$ | $+0.4185$ |

*Interpretation:* On SST, the Hybrid QNN shows a positive empirical trend ($+0.69\%$ accuracy, Cohen's $d = 0.62$), but the 95% confidence interval spans $[-0.70\%, +2.08\%]$ ($p = 0.238$). 

### 2.3 Statistical Limitations of Small Sample Size ($N = 5$)
With only 5 independent random seeds ($df = 4$), the critical $t$-value at $\alpha = 0.05$ is $t_{\text{crit}} = 2.776$. To detect an effect size of $\Delta = 0.007$ as statistically significant at $80\%$ power, a sample size of $N \ge 25$ independent random runs would be required. An honest academic report must present the trend without overclaiming statistical significance.

---

## 3. Quantum Circuit Depth Analysis: 1 Layer vs 2 Layers (Task 4)

| Configuration | CR Accuracy (Mean $\pm$ Std) | CR F1-Score | SST Accuracy (Mean $\pm$ Std) | SST F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Hybrid QNN (1-layer)** | $0.7880 \pm 0.0206$ | $0.7801 \pm 0.0302$ | $0.8078 \pm 0.0045$ | $0.8076 \pm 0.0045$ |
| **Hybrid QNN (2-layer)** | **$0.7943 \pm 0.0061$** | **$0.7931 \pm 0.0069$** | **$0.8105 \pm 0.0032$** | **$0.8103 \pm 0.0033$** |

### Findings:
1. In the initial single-seed ablation run, 1-layer scored 79.95% vs 79.69% for 2-layer on CR. Multi-seed auditing revealed that this was caused by random initialization variance.
2. The 1-layer ansatz exhibits large variance ($\sigma = \pm 0.0206$) on CR, dropping to $76.56\%$ on seeds 7 and 99 due to limited entanglement capacity (CNOT gates applied only once across 4 qubits).
3. The 2-layer ansatz is significantly more stable ($\sigma = \pm 0.0061$ on CR, $\pm 0.0032$ on SST), providing a more reliable parameter landscape for classical gradient descent.

---

## 4. Disaggregated Metric Performance Across Conventions (Task 3)

The table below disaggregates all primary models on the benchmark datasets across Accuracy, Weighted F1, Macro F1, Positive-Class F1, AUC-ROC, and Training Time (saved to `results/disaggregated_metrics.csv`):

| Dataset | Model | Accuracy | Weighted F1 | Macro F1 | Pos-Class F1 | AUC-ROC | Train Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MR** | Logistic Regression | **0.7685** | **0.7685** | **0.7685** | 0.7689 | 0.8372 | 0.5 s |
| MR | Linear SVM (Calibrated) | 0.7516 | 0.7516 | 0.7516 | 0.7516 | 0.8340 | 0.8 s |
| MR | Classical Deep MLP | 0.7545 | 0.7544 | 0.7544 | 0.7510 | 0.8410 | 2.3 s |
| MR | Compact MLP (4-dim) | 0.7610 | 0.7610 | 0.7610 | 0.7632 | 0.8414 | 1.4 s |
| MR | **Hybrid QNN** (4q, 2L) | 0.7591 | 0.7590 | 0.7590 | **0.7644** | **0.8451** | 16.4 s |
| | | | | | | | |
| **CR** | Logistic Regression | 0.7891 | 0.7809 | 0.7562 | **0.8457** | 0.8679 | 0.1 s |
| CR | Linear SVM (Calibrated) | 0.7943 | 0.7919 | 0.7724 | 0.8429 | 0.8672 | 0.1 s |
| CR | Classical Deep MLP | 0.7891 | 0.7889 | 0.7713 | 0.8350 | 0.8533 | 0.6 s |
| CR | Compact MLP (4-dim) | 0.7943 | 0.7927 | 0.7740 | 0.8417 | **0.8735** | 0.3 s |
| CR | **Hybrid QNN** (4q, 2L) | **0.7969** | **0.7962** | **0.7787** | 0.8421 | 0.8720 | 5.7 s |
| | | | | | | | |
| **SST** | Logistic Regression | 0.7930 | 0.7927 | 0.7927 | 0.7996 | 0.8835 | 0.7 s |
| SST | Linear SVM (Calibrated) | 0.8018 | 0.8015 | 0.8015 | 0.8083 | 0.8846 | 0.3 s |
| SST | Classical Deep MLP | 0.8034 | 0.8031 | 0.8031 | 0.7947 | 0.8840 | 1.9 s |
| SST | Compact MLP (4-dim) | 0.8149 | 0.8149 | 0.8149 | 0.8183 | 0.8944 | 0.7 s |
| SST | **Hybrid QNN** (4q, 2L) | **0.8188** | **0.8188** | **0.8188** | **0.8197** | **0.8956** | 18.1 s |
| | | | | | | | |
| **SUBJ** | Logistic Regression | 0.8920 | 0.8921 | 0.8919 | 0.8896 | 0.9555 | 0.5 s |
| SUBJ | Linear SVM (Calibrated) | 0.9050 | 0.9050 | 0.9049 | 0.9018 | 0.9617 | 0.3 s |
| SUBJ | Classical Deep MLP | **0.9110** | **0.9110** | **0.9110** | **0.9093** | 0.9662 | 1.6 s |
| SUBJ | Compact MLP (4-dim) | 0.9090 | 0.9090 | 0.9090 | 0.9072 | **0.9675** | 0.7 s |
| SUBJ | **Hybrid QNN** (4q, 2L) | 0.8990 | 0.8990 | 0.8990 | 0.8977 | 0.9623 | 15.9 s |
| | | | | | | | |
| **MPQA\*** | Logistic Regression | 0.8197 | 0.8010 | 0.7542 | 0.6274 | 0.8779 | 0.7 s |
| MPQA\* | Linear SVM (Calibrated) | 0.8583 | 0.8528 | 0.8242 | 0.7467 | 0.8864 | 0.4 s |
| MPQA\* | Classical Deep MLP | **0.8638** | **0.8609** | **0.8358** | **0.7680** | **0.8931** | 1.8 s |
| MPQA\* | Compact MLP (4-dim) | 0.8620 | 0.8579 | 0.8314 | 0.7596 | 0.8929 | 0.8 s |
| MPQA\* | **Hybrid QNN** (4q, 2L) | 0.8611 | 0.8567 | 0.8298 | 0.7568 | 0.8908 | 17.2 s |

*\*MPQA is flagged with a caveat: 271 sentences overlap between splits due to phrase-level extraction. It is retained solely for completeness with the original repository's claims.*

---

## 5. Final Research Conclusions for Academic Defense

1. **Role of the Classical Bottleneck ($5000 \to 4$):**
   - In hybrid quantum-classical NLP, the initial linear layer compresses high-dimensional TF-IDF vectors into 4 scalar values to match the qubit count.
   - Experiments confirm that this classical compression layer performs the majority of the feature extraction. When the VQC is replaced by a direct classical linear classifier (`Compact MLP`), performance drops by only $\approx 0.7\%$ on SST and is actually $\approx 0.5\%$ higher on CR.
2. **Contribution of the Variational Quantum Circuit:**
   - The IQP embedding and parameter-rich rotation/entanglement layers provide a bounded non-linear transformation in Hilbert space. On larger text datasets (SST), it provides a modest empirical improvement over linear bottleneck representations ($81.05\%$ vs $80.36\%$).
3. **Absence of Quantum Advantage:**
   - There is **no quantum computational or statistical advantage** in this benchmark. The classical baselines (Logistic Regression, SVM, Compact MLP) achieve comparable accuracy while training in a fraction of a second.
4. **Simulator vs. Physical NISQ Constraints:**
   - All quantum circuits were simulated on CPU via classical linear algebra (`default.qubit`). Physical quantum processors would introduce gate infidelity, state preparation and measurement (SPAM) noise, and barren plateau effects that could further degrade performance.
