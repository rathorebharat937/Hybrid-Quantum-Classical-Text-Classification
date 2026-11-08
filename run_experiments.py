#!/usr/bin/env python3
"""
Master Experiment Runner for Quantum Neural Network Text Classification
=======================================================================
Executes all phases: data loading, QNN training, classical baselines,
evaluation, ablation studies, visualization, and report generation.
"""

import os, sys, json, time, csv, traceback, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, 'predictions')
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

for d in [RESULTS_DIR, FIGURES_DIR, PREDICTIONS_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
log_path = os.path.join(LOGS_DIR, f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w')
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_path)

print("=" * 70)
print("QUANTUM NEURAL NETWORK TEXT CLASSIFICATION - EXPERIMENT RUNNER")
print(f"Started: {datetime.now().isoformat()}")
print(f"Device: CPU (macOS {os.uname().machine})")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────
# Import project modules
# ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, PROJECT_ROOT)
from src.data import load_text_dataset
from src.circuit import build_qlayer

# ──────────────────────────────────────────────────────────────────────
# Model Definitions
# ──────────────────────────────────────────────────────────────────────
class HybridQNNClassifier(nn.Module):
    """Hybrid Quantum-Classical Classifier (original architecture)."""
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
    """Classical MLP baseline with same dimensionality."""
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
    """MLP with 4-dim bottleneck matching QNN compression."""
    def __init__(self, input_dim=5000, n_compressed=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, n_compressed)
        self.fc2 = nn.Linear(n_compressed, 2)
    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x

# ──────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute all required metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }
    if y_prob is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auc_roc'] = float('nan')
    else:
        metrics['auc_roc'] = float('nan')
    return metrics, cm

def prepare_tfidf(X_train, X_val, X_test, max_features=5000):
    """TF-IDF vectorization."""
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_tr = vectorizer.fit_transform(list(X_train)).toarray()
    X_va = vectorizer.transform(list(X_val)).toarray()
    X_te = vectorizer.transform(list(X_test)).toarray()
    return X_tr, X_va, X_te, vectorizer

# ──────────────────────────────────────────────────────────────────────
# PyTorch model trainer
# ──────────────────────────────────────────────────────────────────────
def train_pytorch_model(model, X_train, y_train, X_val, y_val,
                        lr=0.001, batch_size=32, n_epochs=10, model_name="Model"):
    """Train a PyTorch model and return training history."""
    device = torch.device('cpu')
    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.long).to(device)
    X_va = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_va = torch.tensor(y_val, dtype=torch.long).to(device)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_acc': [], 'val_f1': []}
    best_val_f1 = 0.0
    best_epoch = 0
    best_state = None

    print(f"\n  Training {model_name} for {n_epochs} epochs...")
    t_start = time.time()

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(X_tr.size(0))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, X_tr.size(0), batch_size):
            idx = perm[i:i + batch_size]
            bx, by = X_tr[idx], y_tr[idx]
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history['train_loss'].append(avg_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            va_out = model(X_va)
            va_pred = torch.argmax(va_out, dim=1).cpu().numpy()
            va_acc = accuracy_score(y_val, va_pred)
            va_f1 = f1_score(y_val, va_pred, average='weighted')

        history['val_acc'].append(va_acc)
        history['val_f1'].append(va_f1)
        print(f"    Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f} | Val Acc: {va_acc:.4f} | Val F1: {va_f1:.4f}")

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.time() - t_start
    print(f"  Best epoch: {best_epoch} (Val F1: {best_val_f1:.4f}) | Training time: {train_time:.1f}s")

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model, history, train_time, best_state

def evaluate_pytorch_model(model, X_test, y_test):
    """Evaluate a PyTorch model on the test set."""
    device = torch.device('cpu')
    X_te = torch.tensor(X_test, dtype=torch.float32).to(device)
    model = model.to(device)
    model.eval()

    t_start = time.time()
    with torch.no_grad():
        out = model(X_te)
        pred = torch.argmax(out, dim=1).cpu().numpy()
        prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
    eval_time = time.time() - t_start

    metrics, cm = compute_metrics(y_test, pred, prob)
    metrics['eval_time'] = eval_time
    return metrics, cm, pred, prob

# ──────────────────────────────────────────────────────────────────────
# sklearn model trainer
# ──────────────────────────────────────────────────────────────────────
def train_eval_sklearn(model_class, model_name, X_train, y_train, X_test, y_test, **kwargs):
    """Train and evaluate an sklearn model."""
    print(f"\n  Training {model_name}...")
    t_start = time.time()
    model = model_class(**kwargs)
    model.fit(X_train, y_train)
    train_time = time.time() - t_start

    t_start = time.time()
    pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        scores = model.decision_function(X_test)
        prob = 1 / (1 + np.exp(-scores))  # sigmoid
    else:
        prob = None
    eval_time = time.time() - t_start

    metrics, cm = compute_metrics(y_test, pred, prob)
    metrics['train_time'] = train_time
    metrics['eval_time'] = eval_time
    print(f"  {model_name}: Acc={metrics['accuracy']:.4f} F1={metrics['f1_score']:.4f} AUC={metrics['auc_roc']:.4f} Time={train_time:.1f}s")
    return metrics, cm, pred, prob

# ──────────────────────────────────────────────────────────────────────
# Visualization functions
# ──────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, title, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_training_curves(history, title, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-o', markersize=4)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Training Loss'); ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, history['val_acc'], 'g-o', markersize=4, label='Val Accuracy')
    ax2.plot(epochs, history['val_f1'], 'r-s', markersize=4, label='Val F1')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Score'); ax2.set_title('Validation Metrics')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_roc_curve(y_true, y_prob, title, path):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

# ──────────────────────────────────────────────────────────────────────
# PHASE 3: Dataset Loading and Statistics
# ──────────────────────────────────────────────────────────────────────
def load_all_datasets():
    """Load all available datasets and compute statistics."""
    datasets = {}
    stats_rows = []
    # Datasets that use local files only (no network needed)
    local_datasets = ['MR', 'CR', 'SST']
    # Datasets that need HuggingFace download
    hf_datasets = ['SUBJ', 'MPQA']

    for name in local_datasets + hf_datasets:
        print(f"\n{'='*50}")
        print(f"Loading dataset: {name}")
        print('='*50)
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = load_text_dataset(name)
            y_train = np.asarray(y_train)
            y_val = np.asarray(y_val)
            y_test = np.asarray(y_test)

            total = len(y_train) + len(y_val) + len(y_test)
            pos_train = int((y_train == 1).sum())
            neg_train = int((y_train == 0).sum())
            pos_test = int((y_test == 1).sum())
            neg_test = int((y_test == 0).sum())

            datasets[name] = {
                'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
                'y_train': y_train, 'y_val': y_val, 'y_test': y_test
            }
            stats_rows.append({
                'dataset': name, 'total': total,
                'train': len(y_train), 'val': len(y_val), 'test': len(y_test),
                'train_pos': pos_train, 'train_neg': neg_train,
                'test_pos': pos_test, 'test_neg': neg_test,
                'pos_ratio': f"{(pos_train+pos_test)/(len(y_train)+len(y_test)):.2%}"
            })
            print(f"  ✅ Loaded: {total} samples (train={len(y_train)}, val={len(y_val)}, test={len(y_test)})")
            print(f"     Train: pos={pos_train}, neg={neg_train} | Test: pos={pos_test}, neg={neg_test}")
        except Exception as e:
            print(f"  ❌ FAILED to load {name}: {e}")
            traceback.print_exc()

    # Save statistics
    if stats_rows:
        df = pd.DataFrame(stats_rows)
        df.to_csv(os.path.join(RESULTS_DIR, 'dataset_statistics.csv'), index=False)
        print(f"\n📊 Dataset statistics saved to results/dataset_statistics.csv")

    # Plot class distributions
    if stats_rows:
        fig, axes = plt.subplots(1, len(stats_rows), figsize=(4*len(stats_rows), 4))
        if len(stats_rows) == 1:
            axes = [axes]
        for ax, row in zip(axes, stats_rows):
            ax.bar(['Positive', 'Negative'],
                   [row['train_pos']+row['test_pos'], row['train_neg']+row['test_neg']],
                   color=['#2196F3', '#FF5722'])
            ax.set_title(f"{row['dataset']}")
            ax.set_ylabel('Count')
        plt.suptitle('Dataset Class Distribution', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'dataset_class_distribution.png'), dpi=150)
        plt.close()
        print(f"📊 Class distribution plot saved")

    return datasets

# ──────────────────────────────────────────────────────────────────────
# Main Experiment Runner
# ──────────────────────────────────────────────────────────────────────
def run_all_experiments():
    """Execute all experiments across all datasets."""
    SEED = 42
    set_seed(SEED)

    # Phase 3: Load datasets
    print("\n" + "="*70)
    print("PHASE 3: DATASET PREPARATION")
    print("="*70)
    datasets = load_all_datasets()

    if not datasets:
        print("❌ No datasets loaded. Exiting.")
        return

    # Phase 5-8: Run experiments
    print("\n" + "="*70)
    print("PHASES 5-8: MODEL TRAINING & EVALUATION")
    print("="*70)

    all_results = []       # For model_comparison.csv
    all_histories = {}     # For training curve plots
    all_roc_data = {}      # For ROC plots
    experiment_summary = {
        'environment': {
            'os': 'macOS 15.5',
            'arch': 'Apple Silicon (arm64)',
            'python': '3.14.6',
            'pennylane': '0.45.1',
            'pytorch': '2.15.0',
            'device': 'cpu',
            'ram_gb': 8,
            'seed': SEED,
        },
        'experiments': [],
        'started_at': datetime.now().isoformat(),
    }

    for ds_name, ds_data in datasets.items():
        print(f"\n{'#'*70}")
        print(f"# DATASET: {ds_name}")
        print(f"{'#'*70}")

        X_train = ds_data['X_train']
        X_val   = ds_data['X_val']
        X_test  = ds_data['X_test']
        y_train = ds_data['y_train']
        y_val   = ds_data['y_val']
        y_test  = ds_data['y_test']

        # TF-IDF preprocessing (shared across all models for this dataset)
        print("\n  ⏳ TF-IDF preprocessing...")
        t_preprocess = time.time()
        X_tr_tfidf, X_va_tfidf, X_te_tfidf, vectorizer = prepare_tfidf(
            X_train, X_val, X_test, max_features=5000
        )
        preprocess_time = time.time() - t_preprocess
        print(f"  Preprocessing time: {preprocess_time:.1f}s | Feature dim: {X_tr_tfidf.shape[1]}")

        # ── Model 1: Logistic Regression ──────────────────────────────
        try:
            set_seed(SEED)
            m, cm, pred, prob = train_eval_sklearn(
                LogisticRegression, "Logistic Regression",
                X_tr_tfidf, y_train, X_te_tfidf, y_test,
                max_iter=1000, random_state=SEED
            )
            all_results.append({**m, 'dataset': ds_name, 'model': 'Logistic Regression'})
            plot_confusion_matrix(cm, f"Logistic Regression - {ds_name}",
                                 os.path.join(FIGURES_DIR, f'{ds_name}_LR_cm.png'))
            if prob is not None:
                plot_roc_curve(y_test, prob, f"Logistic Regression ROC - {ds_name}",
                               os.path.join(FIGURES_DIR, f'{ds_name}_LR_roc.png'))
                all_roc_data[f'{ds_name}_LR'] = (y_test, prob)
            np.savez(os.path.join(PREDICTIONS_DIR, f'{ds_name}_LR_preds.npz'),
                     y_true=y_test, y_pred=pred, y_prob=prob if prob is not None else [])
        except Exception as e:
            print(f"  ❌ Logistic Regression failed on {ds_name}: {e}")

        # ── Model 2: Linear SVM ───────────────────────────────────────
        try:
            set_seed(SEED)
            # Use CalibratedClassifierCV to get probabilities from SVM
            base_svm = LinearSVC(max_iter=2000, random_state=SEED)
            cal_svm = CalibratedClassifierCV(base_svm, cv=3)
            m, cm, pred, prob = train_eval_sklearn(
                lambda **kw: cal_svm, "Linear SVM (Calibrated)",
                X_tr_tfidf, y_train, X_te_tfidf, y_test
            )
            all_results.append({**m, 'dataset': ds_name, 'model': 'Linear SVM'})
            plot_confusion_matrix(cm, f"Linear SVM - {ds_name}",
                                 os.path.join(FIGURES_DIR, f'{ds_name}_SVM_cm.png'))
            if prob is not None:
                plot_roc_curve(y_test, prob, f"Linear SVM ROC - {ds_name}",
                               os.path.join(FIGURES_DIR, f'{ds_name}_SVM_roc.png'))
                all_roc_data[f'{ds_name}_SVM'] = (y_test, prob)
            np.savez(os.path.join(PREDICTIONS_DIR, f'{ds_name}_SVM_preds.npz'),
                     y_true=y_test, y_pred=pred, y_prob=prob if prob is not None else [])
        except Exception as e:
            print(f"  ❌ Linear SVM failed on {ds_name}: {e}")

        # ── Model 3: Classical MLP ────────────────────────────────────
        try:
            set_seed(SEED)
            mlp = ClassicalMLP(input_dim=X_tr_tfidf.shape[1], hidden_dim=64)
            mlp, hist, train_time, best_state = train_pytorch_model(
                mlp, X_tr_tfidf, y_train, X_va_tfidf, y_val,
                lr=0.001, batch_size=32, n_epochs=10, model_name=f"Classical MLP [{ds_name}]"
            )
            m, cm, pred, prob = evaluate_pytorch_model(mlp, X_te_tfidf, y_test)
            m['train_time'] = train_time
            all_results.append({**m, 'dataset': ds_name, 'model': 'Classical MLP'})
            all_histories[f'{ds_name}_MLP'] = hist
            plot_confusion_matrix(cm, f"Classical MLP - {ds_name}",
                                 os.path.join(FIGURES_DIR, f'{ds_name}_MLP_cm.png'))
            plot_training_curves(hist, f"Classical MLP Training - {ds_name}",
                                os.path.join(FIGURES_DIR, f'{ds_name}_MLP_curves.png'))
            if prob is not None:
                plot_roc_curve(y_test, prob, f"Classical MLP ROC - {ds_name}",
                               os.path.join(FIGURES_DIR, f'{ds_name}_MLP_roc.png'))
                all_roc_data[f'{ds_name}_MLP'] = (y_test, prob)
            np.savez(os.path.join(PREDICTIONS_DIR, f'{ds_name}_MLP_preds.npz'),
                     y_true=y_test, y_pred=pred, y_prob=prob if prob is not None else [])
            if best_state:
                torch.save(best_state, os.path.join(CHECKPOINTS_DIR, f'{ds_name}_MLP_best.pt'))
        except Exception as e:
            print(f"  ❌ Classical MLP failed on {ds_name}: {e}")
            traceback.print_exc()

        # ── Model 4: Compact MLP (4-dim bottleneck, no quantum) ──────
        try:
            set_seed(SEED)
            cmlp = CompactMLP(input_dim=X_tr_tfidf.shape[1], n_compressed=4)
            cmlp, hist, train_time, best_state = train_pytorch_model(
                cmlp, X_tr_tfidf, y_train, X_va_tfidf, y_val,
                lr=0.001, batch_size=32, n_epochs=10, model_name=f"Compact MLP (4-dim) [{ds_name}]"
            )
            m, cm, pred, prob = evaluate_pytorch_model(cmlp, X_te_tfidf, y_test)
            m['train_time'] = train_time
            all_results.append({**m, 'dataset': ds_name, 'model': 'Compact MLP (4-dim)'})
            plot_confusion_matrix(cm, f"Compact MLP - {ds_name}",
                                 os.path.join(FIGURES_DIR, f'{ds_name}_CompactMLP_cm.png'))
            np.savez(os.path.join(PREDICTIONS_DIR, f'{ds_name}_CompactMLP_preds.npz'),
                     y_true=y_test, y_pred=pred, y_prob=prob if prob is not None else [])
        except Exception as e:
            print(f"  ❌ Compact MLP failed on {ds_name}: {e}")

        # ── Model 5: Hybrid QNN ───────────────────────────────────────
        try:
            set_seed(SEED)
            qnn = HybridQNNClassifier(input_dim=X_tr_tfidf.shape[1], n_qubits=4, n_layers=2)
            qnn, hist, train_time, best_state = train_pytorch_model(
                qnn, X_tr_tfidf, y_train, X_va_tfidf, y_val,
                lr=0.001, batch_size=32, n_epochs=10, model_name=f"Hybrid QNN [{ds_name}]"
            )
            m, cm, pred, prob = evaluate_pytorch_model(qnn, X_te_tfidf, y_test)
            m['train_time'] = train_time
            all_results.append({**m, 'dataset': ds_name, 'model': 'Hybrid QNN'})
            all_histories[f'{ds_name}_QNN'] = hist
            plot_confusion_matrix(cm, f"Hybrid QNN - {ds_name}",
                                 os.path.join(FIGURES_DIR, f'{ds_name}_QNN_cm.png'))
            plot_training_curves(hist, f"Hybrid QNN Training - {ds_name}",
                                os.path.join(FIGURES_DIR, f'{ds_name}_QNN_curves.png'))
            if prob is not None:
                plot_roc_curve(y_test, prob, f"Hybrid QNN ROC - {ds_name}",
                               os.path.join(FIGURES_DIR, f'{ds_name}_QNN_roc.png'))
                all_roc_data[f'{ds_name}_QNN'] = (y_test, prob)
            np.savez(os.path.join(PREDICTIONS_DIR, f'{ds_name}_QNN_preds.npz'),
                     y_true=y_test, y_pred=pred, y_prob=prob if prob is not None else [])
            if best_state:
                torch.save(best_state, os.path.join(CHECKPOINTS_DIR, f'{ds_name}_QNN_best.pt'))

            experiment_summary['experiments'].append({
                'dataset': ds_name,
                'model': 'Hybrid QNN',
                'n_qubits': 4, 'n_layers': 2,
                'accuracy': m['accuracy'], 'f1_score': m['f1_score'],
                'auc_roc': m['auc_roc'], 'train_time': train_time,
            })
        except Exception as e:
            print(f"  ❌ Hybrid QNN failed on {ds_name}: {e}")
            traceback.print_exc()

    # ── Save comparison CSV ───────────────────────────────────────────
    if all_results:
        cols = ['dataset', 'model', 'accuracy', 'precision', 'recall', 'f1_score',
                'auc_roc', 'train_time', 'eval_time', 'tp', 'tn', 'fp', 'fn']
        df = pd.DataFrame(all_results)
        for c in cols:
            if c not in df.columns:
                df[c] = ''
        df[cols].to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False)
        print(f"\n📊 Model comparison saved to results/model_comparison.csv")

    # ── Comparison bar charts ─────────────────────────────────────────
    if all_results:
        df = pd.DataFrame(all_results)
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
            try:
                fig, ax = plt.subplots(figsize=(max(10, len(datasets)*3), 5))
                pivot = df.pivot_table(index='dataset', columns='model', values=metric)
                pivot.plot(kind='bar', ax=ax, rot=0)
                ax.set_title(f'{metric.replace("_"," ").title()} Comparison Across Datasets')
                ax.set_ylabel(metric.replace("_"," ").title())
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.set_ylim(0, 1.05)
                ax.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(FIGURES_DIR, f'comparison_{metric}.png'), dpi=150, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"  Warning: Could not generate {metric} chart: {e}")

        # Training time comparison
        try:
            fig, ax = plt.subplots(figsize=(max(10, len(datasets)*3), 5))
            pivot = df.pivot_table(index='dataset', columns='model', values='train_time')
            pivot.plot(kind='bar', ax=ax, rot=0)
            ax.set_title('Training Time Comparison (seconds)')
            ax.set_ylabel('Time (s)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, 'comparison_training_time.png'), dpi=150, bbox_inches='tight')
            plt.close()
        except Exception:
            pass

        # Overall model performance comparison (average across datasets)
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            avg_metrics = df.groupby('model')[['accuracy', 'f1_score', 'auc_roc']].mean()
            avg_metrics.plot(kind='bar', ax=ax, rot=15)
            ax.set_title('Average Model Performance Across All Datasets')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1.05)
            ax.legend(title='Metric')
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, 'overall_model_comparison.png'), dpi=150)
            plt.close()
        except Exception:
            pass

    # ── Phase 10: Ablation Studies ────────────────────────────────────
    print("\n" + "="*70)
    print("PHASE 10: ABLATION STUDIES")
    print("="*70)

    ablation_results = []
    # Pick smallest dataset for ablation to keep runtime manageable
    ablation_ds = None
    ablation_ds_name = None
    for name in ['CR', 'MR', 'SUBJ', 'SST', 'MPQA']:
        if name in datasets:
            ablation_ds = datasets[name]
            ablation_ds_name = name
            break

    if ablation_ds:
        X_tr_ab, X_va_ab, X_te_ab, _ = prepare_tfidf(
            ablation_ds['X_train'], ablation_ds['X_val'], ablation_ds['X_test'], max_features=5000
        )
        y_tr_ab = ablation_ds['y_train']
        y_va_ab = ablation_ds['y_val']
        y_te_ab = ablation_ds['y_test']

        # Experiment A: Number of qubits
        print(f"\n  Ablation A: Qubit count on {ablation_ds_name}")
        for nq in [2, 4, 6]:
            try:
                set_seed(SEED)
                print(f"\n    --- {nq} qubits ---")
                qnn_ab = HybridQNNClassifier(input_dim=X_tr_ab.shape[1], n_qubits=nq, n_layers=2)
                qnn_ab, hist, t_time, _ = train_pytorch_model(
                    qnn_ab, X_tr_ab, y_tr_ab, X_va_ab, y_va_ab,
                    lr=0.001, batch_size=32, n_epochs=10, model_name=f"QNN {nq}q"
                )
                m, cm, _, _ = evaluate_pytorch_model(qnn_ab, X_te_ab, y_te_ab)
                m['train_time'] = t_time
                ablation_results.append({
                    'experiment': 'qubit_count', 'parameter': nq,
                    'dataset': ablation_ds_name,
                    'accuracy': m['accuracy'], 'f1_score': m['f1_score'],
                    'auc_roc': m['auc_roc'], 'train_time': t_time
                })
            except Exception as e:
                print(f"    ❌ Failed with {nq} qubits: {e}")

        # Experiment B: Circuit depth
        print(f"\n  Ablation B: Circuit depth on {ablation_ds_name}")
        for nl in [1, 2, 3]:
            try:
                set_seed(SEED)
                print(f"\n    --- {nl} layer(s) ---")
                qnn_ab = HybridQNNClassifier(input_dim=X_tr_ab.shape[1], n_qubits=4, n_layers=nl)
                qnn_ab, hist, t_time, _ = train_pytorch_model(
                    qnn_ab, X_tr_ab, y_tr_ab, X_va_ab, y_va_ab,
                    lr=0.001, batch_size=32, n_epochs=10, model_name=f"QNN {nl}L"
                )
                m, cm, _, _ = evaluate_pytorch_model(qnn_ab, X_te_ab, y_te_ab)
                m['train_time'] = t_time
                ablation_results.append({
                    'experiment': 'circuit_depth', 'parameter': nl,
                    'dataset': ablation_ds_name,
                    'accuracy': m['accuracy'], 'f1_score': m['f1_score'],
                    'auc_roc': m['auc_roc'], 'train_time': t_time
                })
            except Exception as e:
                print(f"    ❌ Failed with {nl} layers: {e}")

        # Experiment C: Learning rate
        print(f"\n  Ablation C: Learning rate on {ablation_ds_name}")
        for lr_val in [0.001, 0.0005, 0.0001]:
            try:
                set_seed(SEED)
                print(f"\n    --- lr={lr_val} ---")
                qnn_ab = HybridQNNClassifier(input_dim=X_tr_ab.shape[1], n_qubits=4, n_layers=2)
                qnn_ab, hist, t_time, _ = train_pytorch_model(
                    qnn_ab, X_tr_ab, y_tr_ab, X_va_ab, y_va_ab,
                    lr=lr_val, batch_size=32, n_epochs=10, model_name=f"QNN lr={lr_val}"
                )
                m, cm, _, _ = evaluate_pytorch_model(qnn_ab, X_te_ab, y_te_ab)
                m['train_time'] = t_time
                ablation_results.append({
                    'experiment': 'learning_rate', 'parameter': lr_val,
                    'dataset': ablation_ds_name,
                    'accuracy': m['accuracy'], 'f1_score': m['f1_score'],
                    'auc_roc': m['auc_roc'], 'train_time': t_time
                })
            except Exception as e:
                print(f"    ❌ Failed with lr={lr_val}: {e}")

    if ablation_results:
        pd.DataFrame(ablation_results).to_csv(
            os.path.join(RESULTS_DIR, 'ablation_study.csv'), index=False
        )
        print(f"\n📊 Ablation results saved to results/ablation_study.csv")

        # Ablation plots
        abl_df = pd.DataFrame(ablation_results)
        for exp_name in abl_df['experiment'].unique():
            sub = abl_df[abl_df['experiment'] == exp_name]
            fig, ax = plt.subplots(figsize=(8, 5))
            x = [str(p) for p in sub['parameter']]
            ax.bar(np.arange(len(x))-0.15, sub['accuracy'], 0.3, label='Accuracy', color='#2196F3')
            ax.bar(np.arange(len(x))+0.15, sub['f1_score'], 0.3, label='F1-Score', color='#FF9800')
            ax.set_xticks(range(len(x)))
            ax.set_xticklabels(x)
            ax.set_xlabel(exp_name.replace('_', ' ').title())
            ax.set_ylabel('Score')
            ax.set_title(f'Ablation: {exp_name.replace("_", " ").title()}')
            ax.legend()
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, f'ablation_{exp_name}.png'), dpi=150)
            plt.close()

    # ── Save experiment summary ───────────────────────────────────────
    experiment_summary['finished_at'] = datetime.now().isoformat()
    if all_results:
        df = pd.DataFrame(all_results)
        experiment_summary['best_accuracy'] = {
            'model': df.loc[df['accuracy'].idxmax(), 'model'],
            'dataset': df.loc[df['accuracy'].idxmax(), 'dataset'],
            'value': float(df['accuracy'].max())
        }
        experiment_summary['best_f1'] = {
            'model': df.loc[df['f1_score'].idxmax(), 'model'],
            'dataset': df.loc[df['f1_score'].idxmax(), 'dataset'],
            'value': float(df['f1_score'].max())
        }
        auc_valid = df.dropna(subset=['auc_roc'])
        if len(auc_valid) > 0:
            experiment_summary['best_auc'] = {
                'model': auc_valid.loc[auc_valid['auc_roc'].idxmax(), 'model'],
                'dataset': auc_valid.loc[auc_valid['auc_roc'].idxmax(), 'dataset'],
                'value': float(auc_valid['auc_roc'].max())
            }
        train_time_valid = df.dropna(subset=['train_time'])
        if len(train_time_valid) > 0:
            experiment_summary['fastest_model'] = {
                'model': train_time_valid.loc[train_time_valid['train_time'].idxmin(), 'model'],
                'dataset': train_time_valid.loc[train_time_valid['train_time'].idxmin(), 'dataset'],
                'time_seconds': float(train_time_valid['train_time'].min())
            }

    with open(os.path.join(RESULTS_DIR, 'experimental_summary.json'), 'w') as f:
        json.dump(experiment_summary, f, indent=2, default=str)
    print(f"\n📊 Experimental summary saved to results/experimental_summary.json")

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED")
    print(f"Finished: {datetime.now().isoformat()}")
    print("="*70)

    # Final results table
    if all_results:
        df = pd.DataFrame(all_results)
        print("\n" + "="*70)
        print("FINAL RESULTS SUMMARY")
        print("="*70)
        for ds in df['dataset'].unique():
            print(f"\n--- {ds} ---")
            sub = df[df['dataset'] == ds]
            for _, row in sub.iterrows():
                print(f"  {row['model']:30s} | Acc: {row['accuracy']:.4f} | F1: {row['f1_score']:.4f} | "
                      f"AUC: {row.get('auc_roc', 'N/A'):.4f} | Time: {row.get('train_time', 0):.1f}s")

if __name__ == '__main__':
    run_all_experiments()

