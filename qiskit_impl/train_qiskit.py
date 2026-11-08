"""
Qiskit Hybrid Model Training and Evaluation Pipeline
===================================================
Executes end-to-end training of the Qiskit Hybrid Quantum-Classical Classifier:
- Supports benchmark datasets: CR, SST, MR, SUBJ, MPQA
- TF-IDF feature extraction (5,000 features, fitted strictly on X_train)
- Saves best validation checkpoint under results/qiskit/checkpoints/
- Evaluates test set metrics: Accuracy, Precision, Recall, Weighted F1, Macro F1, Pos-F1, AUC-ROC
- Generates confusion matrices, ROC curves, and training curves
- Saves predictions and metadata
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.data import load_text_dataset
from qiskit_impl.qiskit_hybrid_model import QiskitHybridClassifier

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

def compute_metrics(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    acc = accuracy_score(y_true, y_pred)
    prec_w = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_w = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    prec_m = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_m = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    prec_bin = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_bin = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_bin = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_prob) if y_prob is not None else float('nan')
    except Exception:
        auc = float('nan')
        
    return {
        "accuracy": acc,
        "weighted_precision": prec_w,
        "weighted_recall": rec_w,
        "weighted_f1": f1_w,
        "macro_precision": prec_m,
        "macro_recall": rec_m,
        "macro_f1": f1_m,
        "pos_precision": prec_bin,
        "pos_recall": rec_bin,
        "pos_f1": f1_bin,
        "auc_roc": auc,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
    }, cm

def train_qiskit_model(
    dataset_name="CR",
    n_qubits=4,
    n_layers=2,
    max_features=5000,
    lr=0.001,
    batch_size=32,
    n_epochs=10,
    seed=42,
    subset_train_size=None
):
    print("=" * 80)
    print(f"TRAINING QISKIT HYBRID QUANTUM-CLASSICAL MODEL: DATASET = {dataset_name}")
    print(f"Configuration: {n_qubits} Qubits, {n_layers} Layers, Batch Size {batch_size}, LR {lr}, Seed {seed}")
    print("=" * 80)

    set_seed(seed)
    device = torch.device("cpu")

    # 1. Load Data
    X_train, X_val, X_test, y_train, y_val, y_test = load_text_dataset(dataset_name)
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    y_test = np.asarray(y_test)

    # 2. TF-IDF Preprocessing (Fitted strictly on X_train)
    t_prep_start = time.time()
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_tfidf = vectorizer.fit_transform(list(X_train)).toarray()
    X_val_tfidf = vectorizer.transform(list(X_val)).toarray()
    X_test_tfidf = vectorizer.transform(list(X_test)).toarray()
    prep_time = time.time() - t_prep_start
    print(f"TF-IDF fitted in {prep_time:.2f}s | Feature Dimension: {X_train_tfidf.shape[1]}")

    # Subsetting for fast experiments if requested
    if subset_train_size is not None and subset_train_size < len(X_train_tfidf):
        perm_sub = np.random.RandomState(seed).permutation(len(X_train_tfidf))[:subset_train_size]
        X_train_tfidf = X_train_tfidf[perm_sub]
        y_train = y_train[perm_sub]
        print(f"Subsampled training set to {subset_train_size} instances for controlled profiling.")

    # Convert to Tensors
    X_tr_t = torch.tensor(X_train_tfidf, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_va_t = torch.tensor(X_val_tfidf, dtype=torch.float32, device=device)
    X_te_t = torch.tensor(X_test_tfidf, dtype=torch.float32, device=device)

    # 3. Instantiate Qiskit Model
    model = QiskitHybridClassifier(
        input_dim=X_train_tfidf.shape[1],
        n_qubits=n_qubits,
        n_layers=n_layers
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_acc": [], "val_f1": [], "epoch_times": []}
    best_val_f1 = 0.0
    best_epoch = 0
    best_state = None

    t_train_start = time.time()

    for epoch in range(n_epochs):
        t_epoch_start = time.time()
        model.train()
        g = torch.Generator().manual_seed(seed + epoch)
        perm = torch.randperm(X_tr_t.size(0), generator=g)

        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, X_tr_t.size(0), batch_size):
            indices = perm[i:i + batch_size]
            bx = X_tr_t[indices]
            by = y_tr_t[indices]

            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        epoch_time = time.time() - t_epoch_start

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            va_out = model(X_va_t)
            va_pred = torch.argmax(va_out, dim=1).cpu().numpy()
            va_acc = accuracy_score(y_val, va_pred)
            va_f1 = f1_score(y_val, va_pred, average="weighted")

        history["train_loss"].append(float(avg_loss))
        history["val_acc"].append(float(va_acc))
        history["val_f1"].append(float(va_f1))
        history["epoch_times"].append(float(epoch_time))

        print(f"Epoch {epoch+1:02d}/{n_epochs:02d} | Loss: {avg_loss:.4f} | Val Acc: {va_acc:.4f} | Val F1: {va_f1:.4f} | Time: {epoch_time:.2f}s")

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    total_train_time = time.time() - t_train_start
    print(f"\nBest Checkpoint Selected: Epoch {best_epoch} (Val F1: {best_val_f1:.4f}) | Total Training Time: {total_train_time:.2f}s")

    # 4. Save Checkpoint
    ckpt_dir = os.path.join(ROOT_DIR, "results/qiskit/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{dataset_name}_qiskit_best.pt")
    if best_state is not None:
        torch.save(best_state, ckpt_path)
        print(f"Saved Qiskit model checkpoint to: {ckpt_path}")
        model.load_state_dict(best_state)

    # 5. Evaluate Held-out Test Set
    model.eval()
    t_eval_start = time.time()
    with torch.no_grad():
        te_out = model(X_te_t)
        te_pred = torch.argmax(te_out, dim=1).cpu().numpy()
        te_prob = torch.softmax(te_out, dim=1)[:, 1].cpu().numpy()
    eval_time = time.time() - t_eval_start

    metrics, cm = compute_metrics(y_test, te_pred, te_prob)
    metrics["train_time_seconds"] = total_train_time
    metrics["eval_time_seconds"] = eval_time
    metrics["best_epoch"] = best_epoch

    print("\n" + "=" * 80)
    print(f"FINAL TEST SET RESULTS: {dataset_name} (QISKIT HYBRID QNN)")
    print("=" * 80)
    print(f"Accuracy:            {metrics['accuracy']:.4f}")
    print(f"Weighted F1:         {metrics['weighted_f1']:.4f}")
    print(f"Macro F1:            {metrics['macro_f1']:.4f}")
    print(f"Positive-Class F1:   {metrics['pos_f1']:.4f}")
    print(f"AUC-ROC:             {metrics['auc_roc']:.4f}")
    print(f"Evaluation Time:     {eval_time:.3f}s")
    print("\nClassification Report:\n", classification_report(y_test, te_pred, digits=4))

    # 6. Save Predictions & Results Tables
    pred_dir = os.path.join(ROOT_DIR, "results/qiskit/predictions")
    os.makedirs(pred_dir, exist_ok=True)
    pred_path = os.path.join(pred_dir, f"{dataset_name}_qiskit_preds.npz")
    np.savez(pred_path, y_true=y_test, y_pred=te_pred, y_prob=te_prob)
    print(f"Saved test predictions to: {pred_path}")

    # CSV Predictions
    preds_df = pd.DataFrame({
        "sample_index": range(len(y_test)),
        "true_label": y_test,
        "predicted_label": te_pred,
        "probability_pos": te_prob
    })
    preds_csv_path = os.path.join(ROOT_DIR, "results/qiskit/qiskit_predictions.csv")
    preds_df.to_csv(preds_csv_path, index=False)
    print(f"Saved CSV predictions to: {preds_csv_path}")

    # CSV Training Log
    log_df = pd.DataFrame({
        "epoch": range(1, n_epochs + 1),
        "train_loss": history["train_loss"],
        "val_acc": history["val_acc"],
        "val_f1": history["val_f1"],
        "epoch_time_sec": history["epoch_times"]
    })
    log_csv_path = os.path.join(ROOT_DIR, "results/qiskit/qiskit_training_log.csv")
    log_df.to_csv(log_csv_path, index=False)
    print(f"Saved training log to: {log_csv_path}")

    # CSV Summary Model Results
    res_df = pd.DataFrame([{
        "dataset": dataset_name,
        "model": "Hybrid QNN (IBM Qiskit)",
        "accuracy": metrics["accuracy"],
        "precision": metrics["weighted_precision"],
        "recall": metrics["weighted_recall"],
        "weighted_f1": metrics["weighted_f1"],
        "macro_f1": metrics["macro_f1"],
        "pos_f1": metrics["pos_f1"],
        "auc_roc": metrics["auc_roc"],
        "train_time_sec": total_train_time,
        "eval_time_sec": eval_time,
        "best_epoch": best_epoch,
        "total_epochs": n_epochs,
        "seed": seed,
        "active_features": X_train_tfidf.shape[1],
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }])
    res_csv_path = os.path.join(ROOT_DIR, "results/qiskit/qiskit_model_results.csv")
    res_df.to_csv(res_csv_path, index=False)
    print(f"Saved model results to: {res_csv_path}")

    # 7. Generate Research Figures
    fig_dir = os.path.join(ROOT_DIR, "results/qiskit/figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Neg (0)', 'Pos (1)'], yticklabels=['Neg (0)', 'Pos (1)'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Qiskit QNN Confusion Matrix - {dataset_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{dataset_name}_qiskit_cm.png"), dpi=200)
    plt.close()

    # Training Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    epochs_range = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs_range, history["train_loss"], 'b-o', markersize=4, label='Cross-Entropy Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Qiskit Training Loss - {dataset_name}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs_range, history["val_acc"], 'g-s', markersize=4, label='Val Accuracy')
    ax2.plot(epochs_range, history["val_f1"], 'r-^', markersize=4, label='Val F1 (Weighted)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Metric Score')
    ax2.set_title(f'Qiskit Validation Trajectory - {dataset_name}')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{dataset_name}_qiskit_training_curves.png"), dpi=200)
    plt.close()

    # ROC Curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, te_prob)
    roc_auc_val = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, color='#1976D2', lw=2, label=f'ROC Curve (AUC = {roc_auc_val:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Qiskit QNN ROC - {dataset_name}')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{dataset_name}_qiskit_roc.png"), dpi=200)
    plt.close()

    print(f"Generated research figures in: results/qiskit/figures/")

    return metrics, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Qiskit Hybrid Text Classifier")
    parser.add_argument("--dataset", type=str, default="CR", choices=["CR", "SST", "MR", "SUBJ", "MPQA"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", type=int, default=None)
    args = parser.parse_args()

    train_qiskit_model(
        dataset_name=args.dataset,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        subset_train_size=args.subset
    )

