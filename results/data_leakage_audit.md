# Data Leakage & Split Integrity Audit

**Audit Date:** November 2026  
**Auditor:** ML & QML Research Auditor  
**Scope:** MR, CR, SST, SUBJ, MPQA Datasets  

---

## 1. Executive Summary

| Dataset | Split Method | Total Samples | Train/Val/Test | Exact Overlap (Train $\cap$ Test) | Leakage Risk Level |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MR** | Stratified Train-Test Split (seed=42) | 10,662 | 8,530 / 1,065 / 1,067 | 0 (0.00%) | **None (Clean Benchmark)** |
| **CR** | Stratified Train-Test Split (seed=42) | 3,770 | 3,024 / 362 / 384 | 1 (0.03%) | **Minimal (Trivial Sentence)** |
| **SST** | Sentence-Level Benchmark (SetFit/sst2) | 9,613 | 6,920 / 872 / 1,821 | 2 (0.03%) | **Minimal (Short Phrasal)** |
| **SUBJ** | Sentence-Level Benchmark (SetFit/subj) | 10,000 | 8,000 / 1,000 / 1,000 | 0 (0.00%) | **None (Clean Benchmark)** |
| **MPQA** | Stratified 80/10/10 Split on Pooled Corpus | 10,859 | 8,687 / 1,085 / 1,087 | 271 (3.12%) | **High (Documented Limitation — Excluded from Clean Claims)** |

---

## 2. Detailed Findings by Dataset

### 2.1 MR (Movie Reviews)
- **Splits:** 8,530 train, 1,065 val, 1,067 test.
- **Overlap:** Exactly 0 duplicate sentences between train, validation, and test sets.
- **Internal Duplicates:** 0 in train, 0 in val, 0 in test.
- **Class Balance:** Train is precisely balanced (4,265 pos / 4,265 neg); Test is 533 pos / 534 neg.
- **Preprocessing:** TF-IDF vocabulary is fitted strictly on `X_train` (`vectorizer.fit_transform(X_train)`), then used to transform `X_val` and `X_test`. Zero out-of-fold vocabulary leakage occurs.

### 2.2 CR (Customer Reviews)
- **Splits:** 3,024 train, 362 val, 384 test.
- **Overlap:** Exactly 1 overlap between Train and Test:
  - Text: `"i would recommend this camera unequivocally !"` (Label: 1 in both splits).
  - Also 1 overlap between Train and Val.
- **Assessment:** Trivial sentiment phrasing common in product review corpora. No synthetic leakage.

### 2.3 SST (Stanford Sentiment Treebank / SST-2)
- **Benchmark Source Investigation:**
  - In `download_datasets.py`, data was fetched from `SetFit/sst2` (sentences: 6,920 train, 872 dev, 1,821 test), which mirrors the binary sentence split from Socher et al. (2013).
  - Note: The full GLUE SST-2 benchmark contains 67,349 training phrase-level snippets (extracted from treebank parse nodes). The original paper's table cites "65,528 Train / 872 Val / 1,821 Test", which reflects phrase-level training expansions. Our local execution uses sentence-level instances (6,920).
- **Overlap:** 2 short instances between Train and Test (e.g. `"no. ...."`).
- **Internal Duplicates:** 9 entries within training data.
- **Assessment:** Acceptable; reflects natural occurrence in sentiment benchmarks.

### 2.4 SUBJ (Subjectivity Dataset)
- **Splits:** 8,000 train, 1,000 val, 1,000 test.
- **Overlap:** Exactly 0 overlaps across all partitions.
- **Internal Duplicates:** 0.
- **Class Balance:** Train: 4,033 pos / 3,967 neg; Test: 478 pos / 522 neg.
- **Assessment:** Flawless split separation.

### 2.5 MPQA (Deep Investigation & Exclusion Finding)
- **Investigation of Document/Source Identifiers:**
  - The dataset distribution hosted on Hugging Face (`jxm/mpqa`) contains only two columns: `sentence` (string) and `label` (int64). It does **not** include document-level metadata, news article identifiers, or topic tags.
  - The corpus is constructed from short extracted phrases/clauses (e.g., `"approval"`, `"no right"`, `"denounced"`, `"thought"`, `"will not"`, `"to blame..."`).
  - Total samples in the pooled dataset: 10,859. Total unique texts: 8,240. Over 2,600 phrases are repeated multiple times across the corpus.
- **Leakage Cause:**
  - The repository's author pooled `train + dev + test` and applied `train_test_split(test_size=0.1)`. Because phrases repeat across articles, 271 sentences appear in both training and test sets.
  - Furthermore, inspecting the original author splits of `jxm/mpqa` reveals that even the upstream Hugging Face splits contain 262 exact overlaps between Train and Test and 253 between Dev and Test.
- **Auditor Resolution:**
  - Because no document identifiers exist in the feature set, it is impossible to reconstruct a document-isolated split without external corpus realignment.
  - **Academic Recommendation:** MPQA **must be excluded from claims of clean generalization benchmarks**. Its empirical results are retained strictly as a documented limitation of the original paper's codebase and data dispatcher.

---

## 3. Preprocessing & Modeling Protocol Audit

1. **TF-IDF Vocabulary Isolation:**
   - In `run_experiments.py`, line 96:
     ```python
     vectorizer = TfidfVectorizer(max_features=max_features)
     X_tr = vectorizer.fit_transform(list(X_train)).toarray()
     X_va = vectorizer.transform(list(X_val)).toarray()
     X_te = vectorizer.transform(list(X_test)).toarray()
     ```
   - **Confirmed:** Vocabulary and IDF weights are determined solely from `X_train`. Zero test-set signal enters feature extraction.
2. **Hyperparameter Tuning:**
   - Model selection (early checkpoint saving) used `val_f1` on `X_val_tensor`. Test data was strictly loaded once at final inference.
3. **Metric Definitions:**
   - Binary positive class: `1` (positive / subjective).
   - Both weighted and positive-class metrics are explicitly disaggregated in `results/disaggregated_metrics.csv`.
