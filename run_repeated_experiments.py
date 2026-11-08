#!/usr/bin/env python3
"""
Repeated Seeds & Statistical Significance Runner
================================================
Runs 5 independent seeds (42, 123, 2026, 7, 99) on CR and SST datasets.
Compares:
  - Logistic Regression
  - Linear SVM (Calibrated)
  - Classical Deep MLP
  - Compact MLP (4-dim)
  - Hybrid QNN (1-layer)
  - Hybrid QNN (2-layer, default)

Records detailed timing breakdowns and statistical summaries (mean ± std).
"""

import os, sys, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd

warnings.filterwarnings('ignore')

os.environ['HF_HUB_OFFLINE'] = '1'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.data import load_text_dataset
from src.circuit import build_qlayer

SEEDS = [42, 123, 2026, 7, 99]
DATASETS = ['CR', 'SST']

class HybridQNNClassifier(nn.Module):
    def __init__(self, input_dim=5000, n_qubits=4, n_layers=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, n_qubits)
        self.qlayer = build_qlayer(n_qubits=n_qubits, n_layers=n_layers)
        self.fc2 = nn.Linear(n_qubits, 2)
    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.qlayer(x)
        x = self.fc2(x)
        return x

class ClassicalMLP(nn.Module):
    def __init__(self, input_dim=5000, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.net(x)

class CompactMLP(nn.Module):
    def __init__(self, input_dim=5000, n_compressed=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, n_compressed)
        self.fc2 = nn.Linear(n_compressed, 2)
    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def compute_metrics(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob) if y_prob is not None else float('nan')
    except Exception:
        auc = float('nan')
    return {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1,
        'auc_roc': auc, 'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }

def train_and_eval_pytorch(model, X_tr, y_tr, X_va, y_va, X_te, y_te, seed, batch_size=32, n_epochs=10, lr=0.001):
    device = torch.device('cpu')
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long).to(device)
    X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    best_state = None

    t_train_start = time.time()
    for epoch in range(n_epochs):
        model.train()
        g = torch.Generator().manual_seed(seed + epoch)
        perm = torch.randperm(X_tr_t.size(0), generator=g)

        for i in range(0, X_tr_t.size(0), batch_size):
            idx = perm[i:i + batch_size]
            bx, by = X_tr_t[idx], y_tr_t[idx]
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        # Validation timing
        model.eval()
        with torch.no_grad():
            va_out = model(X_va_t)
            va_pred = torch.argmax(va_out, dim=1).cpu().numpy()
            va_f1 = f1_score(y_va, va_pred, average='weighted')

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.time() - t_train_start

    # Evaluate
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    t_eval_start = time.time()
    with torch.no_grad():
        te_out = model(X_te_t)
        te_pred = torch.argmax(te_out, dim=1).cpu().numpy()
        te_prob = torch.softmax(te_out, dim=1)[:, 1].cpu().numpy()
    eval_time = time.time() - t_eval_start

    metrics = compute_metrics(y_te, te_pred, te_prob)
    metrics['train_time'] = train_time
    metrics['eval_time'] = eval_time
    return metrics

def run_seed_experiments():
    all_runs = []
    print("=" * 70)
    print("STARTING 5-SEED REPEATED SCIENTIFIC EXPERIMENTS")
    print(f"Seeds: {SEEDS}")
    print(f"Datasets: {DATASETS}")
    print("=" * 70)

    for ds_name in DATASETS:
        print(f"\n==================== DATASET: {ds_name} ====================")
        # Load dataset with benchmark seed
        X_train, X_val, X_test, y_train, y_val, y_test = load_text_dataset(ds_name)
        
        # TF-IDF fitting strictly on X_train
        t_prep_start = time.time()
        vectorizer = TfidfVectorizer(max_features=5000)
        X_tr_tfidf = vectorizer.fit_transform(list(X_train)).toarray()
        X_va_tfidf = vectorizer.transform(list(X_val)).toarray()
        X_te_tfidf = vectorizer.transform(list(X_test)).toarray()
        preprocess_time = time.time() - t_prep_start
        print(f"TF-IDF fitted in {preprocess_time:.2f}s | Shape: {X_tr_tfidf.shape}")

        for seed in SEEDS:
            print(f"\n--- Running Seed: {seed} ---")

            # 1. Logistic Regression
            set_seed(seed)
            t_tr_start = time.time()
            lr_model = LogisticRegression(max_iter=1000, random_state=seed)
            lr_model.fit(X_tr_tfidf, y_train)
            lr_tr_time = time.time() - t_tr_start

            t_ev_start = time.time()
            lr_pred = lr_model.predict(X_te_tfidf)
            lr_prob = lr_model.predict_proba(X_te_tfidf)[:, 1]
            lr_ev_time = time.time() - t_ev_start

            lr_m = compute_metrics(y_test, lr_pred, lr_prob)
            lr_m.update({
                'dataset': ds_name, 'model': 'Logistic Regression', 'seed': seed,
                'train_time': lr_tr_time, 'eval_time': lr_ev_time, 'preprocess_time': preprocess_time
            })
            all_runs.append(lr_m)
            print(f"  [LR]        Acc: {lr_m['accuracy']:.4f} | F1: {lr_m['f1_score']:.4f} | AUC: {lr_m['auc_roc']:.4f} | Time: {lr_tr_time:.2f}s")

            # 2. Linear SVM (Calibrated)
            set_seed(seed)
            t_tr_start = time.time()
            base_svm = LinearSVC(max_iter=2000, random_state=seed)
            cal_svm = CalibratedClassifierCV(base_svm, cv=3)
            cal_svm.fit(X_tr_tfidf, y_train)
            svm_tr_time = time.time() - t_tr_start

            t_ev_start = time.time()
            svm_pred = cal_svm.predict(X_te_tfidf)
            svm_prob = cal_svm.predict_proba(X_te_tfidf)[:, 1]
            svm_ev_time = time.time() - t_ev_start

            svm_m = compute_metrics(y_test, svm_pred, svm_prob)
            svm_m.update({
                'dataset': ds_name, 'model': 'Linear SVM', 'seed': seed,
                'train_time': svm_tr_time, 'eval_time': svm_ev_time, 'preprocess_time': preprocess_time
            })
            all_runs.append(svm_m)
            print(f"  [SVM]       Acc: {svm_m['accuracy']:.4f} | F1: {svm_m['f1_score']:.4f} | AUC: {svm_m['auc_roc']:.4f} | Time: {svm_tr_time:.2f}s")

            # 3. Classical MLP
            set_seed(seed)
            mlp = ClassicalMLP(input_dim=X_tr_tfidf.shape[1], hidden_dim=64)
            mlp_m = train_and_eval_pytorch(mlp, X_tr_tfidf, y_train, X_va_tfidf, y_val, X_te_tfidf, y_test, seed=seed)
            mlp_m.update({
                'dataset': ds_name, 'model': 'Classical MLP', 'seed': seed, 'preprocess_time': preprocess_time
            })
            all_runs.append(mlp_m)
            print(f"  [MLP]       Acc: {mlp_m['accuracy']:.4f} | F1: {mlp_m['f1_score']:.4f} | AUC: {mlp_m['auc_roc']:.4f} | Time: {mlp_m['train_time']:.2f}s")

            # 4. Compact MLP (4-dim bottleneck)
            set_seed(seed)
            cmlp = CompactMLP(input_dim=X_tr_tfidf.shape[1], n_compressed=4)
            cmlp_m = train_and_eval_pytorch(cmlp, X_tr_tfidf, y_train, X_va_tfidf, y_val, X_te_tfidf, y_test, seed=seed)
            cmlp_m.update({
                'dataset': ds_name, 'model': 'Compact MLP (4-dim)', 'seed': seed, 'preprocess_time': preprocess_time
            })
            all_runs.append(cmlp_m)
            print(f"  [CompactMLP]Acc: {cmlp_m['accuracy']:.4f} | F1: {cmlp_m['f1_score']:.4f} | AUC: {cmlp_m['auc_roc']:.4f} | Time: {cmlp_m['train_time']:.2f}s")

            # 5. Hybrid QNN (1-layer)
            set_seed(seed)
            qnn1 = HybridQNNClassifier(input_dim=X_tr_tfidf.shape[1], n_qubits=4, n_layers=1)
            qnn1_m = train_and_eval_pytorch(qnn1, X_tr_tfidf, y_train, X_va_tfidf, y_val, X_te_tfidf, y_test, seed=seed)
            qnn1_m.update({
                'dataset': ds_name, 'model': 'Hybrid QNN (1-layer)', 'seed': seed, 'preprocess_time': preprocess_time
            })
            all_runs.append(qnn1_m)
            print(f"  [QNN-1L]    Acc: {qnn1_m['accuracy']:.4f} | F1: {qnn1_m['f1_score']:.4f} | AUC: {qnn1_m['auc_roc']:.4f} | Time: {qnn1_m['train_time']:.2f}s")

            # 6. Hybrid QNN (2-layer, default)
            set_seed(seed)
            qnn2 = HybridQNNClassifier(input_dim=X_tr_tfidf.shape[1], n_qubits=4, n_layers=2)
            qnn2_m = train_and_eval_pytorch(qnn2, X_tr_tfidf, y_train, X_va_tfidf, y_val, X_te_tfidf, y_test, seed=seed)
            qnn2_m.update({
                'dataset': ds_name, 'model': 'Hybrid QNN (2-layer)', 'seed': seed, 'preprocess_time': preprocess_time
            })
            all_runs.append(qnn2_m)
            print(f"  [QNN-2L]    Acc: {qnn2_m['accuracy']:.4f} | F1: {qnn2_m['f1_score']:.4f} | AUC: {qnn2_m['auc_roc']:.4f} | Time: {qnn2_m['train_time']:.2f}s")

    # Save all raw runs
    df_runs = pd.DataFrame(all_runs)
    df_runs.to_csv('results/repeated_seed_results.csv', index=False)
    print("\nSaved detailed runs to results/repeated_seed_results.csv")

    # Compute Statistical Summary (Mean ± Std)
    summary_list = []
    metrics_cols = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'train_time', 'eval_time']

    for (ds, model), group in df_runs.groupby(['dataset', 'model']):
        row = {'dataset': ds, 'model': model, 'runs': len(group)}
        for c in metrics_cols:
            mean_v = group[c].mean()
            std_v = group[c].std()
            row[f'{c}_mean'] = mean_v
            row[f'{c}_std'] = std_v
            row[f'{c}_display'] = f"{mean_v:.4f} ± {std_v:.4f}" if c != 'train_time' else f"{mean_v:.2f}s ± {std_v:.2f}s"
        summary_list.append(row)

    df_summary = pd.DataFrame(summary_list)
    df_summary.to_csv('results/repeated_seed_summary.csv', index=False)
    print("Saved statistical summary to results/repeated_seed_summary.csv")

    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY (MEAN ± STD ACROSS 5 SEEDS)")
    print("=" * 80)
    for ds in DATASETS:
        print(f"\n--- {ds} ---")
        sub = df_summary[df_summary['dataset'] == ds]
        for _, r in sub.iterrows():
            print(f"  {r['model']:22s} | Acc: {r['accuracy_display']} | F1: {r['f1_score_display']} | AUC: {r['auc_roc_display']} | Train: {r['train_time_display']}")

if __name__ == '__main__':
    run_seed_experiments()

