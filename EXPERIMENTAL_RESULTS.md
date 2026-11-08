# Experimental Results and Scientific Analysis
## Quantum Neural Network for Text Classification Using Hybrid Quantum-Classical Learning

**Academic Evaluation Report for Final Year University Assignment**  
**Author / Student Researcher:** B.Tech Student (Quantum Computing & AI)  
**Execution Environment:** macOS 15.5, Apple Silicon arm64 (8 CPU Cores, 8 GB Unified Memory)  
**Software Versions:** Python 3.14.6, PennyLane 0.45.1, PyTorch 2.15.0, Scikit-Learn 1.9.1  
**Reproducibility Seed:** 42 (Single run) & Multi-Seed Suite (`[42, 123, 2026, 7, 99]`)  

---

## 1. Benchmark Datasets Overview

The model was experimentally evaluated across five standard binary text classification benchmarks. All splits, class distributions, and sample counts were verified:

| Dataset | Domain / Task | Train | Val | Test | Total | Positive Ratio | Audit Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MR** | Movie Reviews (Rotten Tomatoes) | 8,530 | 1,065 | 1,067 | 10,662 | 49.99% | **Clean Split (0 leaks)** |
| **CR** | Customer Product Reviews | 3,024 | 362 | 384 | 3,770 | 63.79% | **Clean Split (1 trivial leak, 0.03%)** |
| **SST** | Stanford Sentiment Treebank (SST-2) | 6,920 | 872 | 1,821 | 9,613 | 51.70% | **Clean Split (2 phrase leaks, 0.03%)** |
| **SUBJ** | Subjectivity / Objectivity Sentences | 8,000 | 1,000 | 1,000 | 10,000 | 50.12% | **Clean Split (0 leaks)** |
| **MPQA\*** | Opinion Polarity Dataset | 8,687 | 1,085 | 1,087 | 10,859 | 31.57% | **Excluded from Clean Benchmark (271 phrase overlaps)** |

*\*MPQA Note: In accordance with the scientific audit, MPQA contains 271 overlapping phrases between splits due to cross-article sub-clause repetition in `jxm/mpqa`. Its results are retained strictly for comparison with the original author's claims.*

*Figure Reference:* Class balances are plotted in `results/figures/dataset_class_distribution.png`.

---

## 2. Comprehensive Benchmark Results (Disaggregated by Metric Convention)

The table below presents the single-seed (seed=42) test set results across all metric definitions. Data is stored in `results/disaggregated_metrics.csv`:

| Dataset | Model | Accuracy | Weighted F1 | Macro F1 | Pos-Class F1 | AUC-ROC | Train Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MR** | Logistic Regression | **0.7685** | **0.7685** | **0.7685** | 0.7689 | 0.8372 | **0.5 s** |
| MR | Linear SVM (Calibrated) | 0.7516 | 0.7516 | 0.7516 | 0.7516 | 0.8340 | 0.8 s |
| MR | Classical Deep MLP | 0.7545 | 0.7544 | 0.7544 | 0.7510 | 0.8410 | 2.3 s |
| MR | Compact MLP (4-dim bottleneck) | 0.7610 | 0.7610 | 0.7610 | 0.7632 | 0.8414 | 1.4 s |
| MR | **Hybrid QNN** (4-qubit, 2-layer) | 0.7591 | 0.7590 | 0.7590 | **0.7644** | **0.8451** | 16.4 s |
| | | | | | | | |
| **CR** | Logistic Regression | 0.7891 | 0.7809 | 0.7562 | **0.8457** | 0.8679 | **0.1 s** |
| CR | Linear SVM (Calibrated) | 0.7943 | 0.7919 | 0.7724 | 0.8429 | 0.8672 | **0.1 s** |
| CR | Classical Deep MLP | 0.7891 | 0.7889 | 0.7713 | 0.8350 | 0.8533 | 0.6 s |
| CR | Compact MLP (4-dim bottleneck) | 0.7943 | 0.7927 | 0.7740 | 0.8417 | **0.8735** | 0.3 s |
| CR | **Hybrid QNN** (4-qubit, 2-layer) | **0.7969** | **0.7962** | **0.7787** | 0.8421 | 0.8720 | 5.7 s |
| | | | | | | | |
| **SST** | Logistic Regression | 0.7930 | 0.7927 | 0.7927 | 0.7996 | 0.8835 | 0.7 s |
| SST | Linear SVM (Calibrated) | 0.8018 | 0.8015 | 0.8015 | 0.8083 | 0.8846 | **0.3 s** |
| SST | Classical Deep MLP | 0.8034 | 0.8031 | 0.8031 | 0.7947 | 0.8840 | 1.9 s |
| SST | Compact MLP (4-dim bottleneck) | 0.8149 | 0.8149 | 0.8149 | 0.8183 | 0.8944 | 0.7 s |
| SST | **Hybrid QNN** (4-qubit, 2-layer) | **0.8188** | **0.8188** | **0.8188** | **0.8197** | **0.8956** | 18.1 s |
| | | | | | | | |
| **SUBJ** | Logistic Regression | 0.8920 | 0.8921 | 0.8919 | 0.8896 | 0.9555 | 0.5 s |
| SUBJ | Linear SVM (Calibrated) | 0.9050 | 0.9050 | 0.9049 | 0.9018 | 0.9617 | **0.3 s** |
| SUBJ | Classical Deep MLP | **0.9110** | **0.9110** | **0.9110** | **0.9093** | 0.9662 | 1.6 s |
| SUBJ | Compact MLP (4-dim bottleneck) | 0.9090 | 0.9090 | 0.9090 | 0.9072 | **0.9675** | 0.7 s |
| SUBJ | **Hybrid QNN** (4-qubit, 2-layer) | 0.8990 | 0.8990 | 0.8990 | 0.8977 | 0.9623 | 15.9 s |
| | | | | | | | |
| **MPQA\*** | Logistic Regression | 0.8197 | 0.8010 | 0.7542 | 0.6274 | 0.8779 | 0.7 s |
| MPQA\* | Linear SVM (Calibrated) | 0.8583 | 0.8528 | 0.8242 | 0.7467 | 0.8864 | **0.4 s** |
| MPQA\* | Classical Deep MLP | **0.8638** | **0.8609** | **0.8358** | **0.7680** | **0.8931** | 1.8 s |
| MPQA\* | Compact MLP (4-dim bottleneck) | 0.8620 | 0.8579 | 0.8314 | 0.7596 | 0.8929 | 0.8 s |
| MPQA\* | **Hybrid QNN** (4-qubit, 2-layer) | 0.8611 | 0.8567 | 0.8298 | 0.7568 | 0.8908 | 17.2 s |

---

## 3. Multi-Seed Statistical Analysis (5 Seeds: 42, 123, 2026, 7, 99)

To verify whether observed differences are statistically significant, repeated evaluations were executed on CR and SST (data stored in `results/repeated_seed_summary.csv`):

### 3.1 Mean $\pm$ Standard Deviation Across 5 Seeds
| Dataset | Model | Accuracy | 95% Confidence Interval | Weighted F1 | AUC-ROC | Train Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CR** | Logistic Regression | $0.7891 \pm 0.0000$ | $[0.7891, 0.7891]$ | $0.7809 \pm 0.0000$ | $0.8679 \pm 0.0000$ | $0.13\text{s} \pm 0.01\text{s}$ |
| CR | Linear SVM | $0.7943 \pm 0.0000$ | $[0.7943, 0.7943]$ | $0.7919 \pm 0.0000$ | $0.8672 \pm 0.0000$ | $0.07\text{s} \pm 0.01\text{s}$ |
| CR | Classical Deep MLP | $0.7812 \pm 0.0064$ | $[0.7733, 0.7892]$ | $0.7809 \pm 0.0063$ | $0.8565 \pm 0.0086$ | $0.61\text{s} \pm 0.07\text{s}$ |
| CR | Compact MLP (4-dim) | **$0.7990 \pm 0.0050$** | $[0.7927, 0.8052]$ | **$0.7978 \pm 0.0053$** | $0.8724 \pm 0.0006$ | $0.26\text{s} \pm 0.01\text{s}$ |
| CR | Hybrid QNN (1-layer) | $0.7880 \pm 0.0206$ | $[0.7624, 0.8136]$ | $0.7801 \pm 0.0302$ | $0.8713 \pm 0.0037$ | $4.30\text{s} \pm 0.12\text{s}$ |
| CR | Hybrid QNN (2-layer) | $0.7943 \pm 0.0061$ | $[0.7867, 0.8018]$ | $0.7931 \pm 0.0069$ | **$0.8727 \pm 0.0005$** | $5.68\text{s} \pm 0.12\text{s}$ |
| | | | | | | |
| **SST** | Logistic Regression | $0.7930 \pm 0.0000$ | $[0.7930, 0.7930]$ | $0.7927 \pm 0.0000$ | $0.8835 \pm 0.0000$ | $0.67\text{s} \pm 0.11\text{s}$ |
| SST | Linear SVM | $0.8018 \pm 0.0000$ | $[0.8018, 0.8018]$ | $0.8015 \pm 0.0000$ | $0.8846 \pm 0.0000$ | $0.26\text{s} \pm 0.03\text{s}$ |
| SST | Classical Deep MLP | $0.8043 \pm 0.0048$ | $[0.7983, 0.8103]$ | $0.8039 \pm 0.0052$ | $0.8862 \pm 0.0031$ | $1.45\text{s} \pm 0.13\text{s}$ |
| SST | Compact MLP (4-dim) | $0.8036 \pm 0.0085$ | $[0.7931, 0.8142]$ | $0.8033 \pm 0.0085$ | $0.8907 \pm 0.0056$ | $0.64\text{s} \pm 0.07\text{s}$ |
| SST | Hybrid QNN (1-layer) | $0.8078 \pm 0.0045$ | $[0.8022, 0.8134]$ | $0.8076 \pm 0.0045$ | $0.8932 \pm 0.0030$ | $9.65\text{s} \pm 0.23\text{s}$ |
| SST | Hybrid QNN (2-layer) | **$0.8105 \pm 0.0032$** | $[0.8066, 0.8145]$ | **$0.8103 \pm 0.0033$** | **$0.8936 \pm 0.0024$** | $13.00\text{s} \pm 0.43\text{s}$ |

### 3.2 Paired Statistical Significance Testing (QNN vs. Compact MLP)
- **On CR:**
  - Mean accuracy difference: $\Delta = -0.0047$ (SE: $0.0034$)
  - 95% Confidence Interval: $[-0.0142, +0.0048]$
  - Paired $t$-test: $t = -1.3646$, $p = 0.2441$
  - Cohen's $d = -0.61$
  - **Verdict:** No statistically significant difference. Compact classical MLP performs identically or slightly better.
- **On SST:**
  - Mean accuracy difference: $\Delta = +0.0069$ (SE: $0.0050$)
  - 95% Confidence Interval: $[-0.0070, +0.0208]$
  - Paired $t$-test: $t = 1.3850$, $p = 0.2383$
  - Cohen's $d = +0.62$
  - **Verdict:** While the QNN exhibits an empirical gain ($+0.69\%$), the 95% confidence interval crosses zero ($p = 0.238$). A sample size of $N=5$ lacks the power to confirm statistical significance at $\alpha = 0.05$.

---

## 4. Controlled Quantum Ablation Study (CR Dataset)

| Experiment | Tested Parameter | Accuracy | F1-Score | AUC-ROC | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Qubit Scaling** | 2 Qubits | 0.7813 | 0.7697 | **0.8739** | 2.8 s |
| | **4 Qubits (Default)** | **0.7969** | **0.7962** | 0.8720 | 6.7 s |
| | 6 Qubits | **0.7969** | **0.7962** | 0.8733 | 10.6 s |
| | | | | | |
| **Circuit Depth** | 1 Layer | **0.7995** | **0.7967** | **0.8745** | 4.2 s |
| | **2 Layers (Default)** | 0.7969 | 0.7962 | 0.8720 | 5.7 s |
| | 3 Layers | 0.7917 | 0.7906 | 0.8722 | 7.3 s |
| | | | | | |
| **Learning Rate** | **0.001 (Default)** | **0.7969** | **0.7962** | **0.8720** | 5.8 s |
| | 0.0005 | 0.7656 | 0.7420 | 0.8684 | 5.9 s |
| | 0.0001 | 0.6432 | 0.6010 | 0.6390 | 5.6 s |

*Audit Clarification on Circuit Depth:* Multi-seed evaluation confirmed that 1-layer ansatz has high seed-to-seed variance ($\pm 0.0206$), whereas 2-layer ansatz provides lower variance ($\pm 0.0061$) and superior stability on larger corpora (SST).

---

## 5. Final Academic Research Conclusions

1. **Role of the Classical Bottleneck:**
   The hybrid model compresses 5,000 TF-IDF features to 4 dimensions using an initial classical linear layer before feeding them to the 4-qubit circuit. Experimental results demonstrate that this classical compression layer performs the primary dimensionality reduction and feature selection.
2. **Quantum Expressivity vs Classical Baselines:**
   The variational quantum circuit provides a bounded non-linear transformation in Hilbert space. On SST, it achieves an empirical improvement over the classical bottleneck ($81.05\%$ vs $80.36\%$), but the difference does not meet statistical significance at $N=5$ ($p = 0.238$).
3. **Absence of Quantum Advantage:**
   Standard classical models (Logistic Regression, Linear SVM) match or exceed the QNN on several datasets while training $20\times$ to $50\times$ faster. There is **no evidence of practical quantum advantage** in this benchmark setting.
4. **NISQ Realities:**
   Simulating quantum circuits classically on CPU requires $2^n$ matrix-vector products. On real quantum hardware, quantum gate infidelity, decoherence, and shot noise would further attenuate subtle decision boundaries.
