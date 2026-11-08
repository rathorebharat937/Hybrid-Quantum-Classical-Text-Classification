"""
Three-Way Cross-Framework Benchmark and Visualization Script
============================================================
Synthesizes verified experimental benchmarks across:
1. Classical Baselines (Logistic Regression, Linear SVM, Deep MLP, Compact MLP)
2. PennyLane Hybrid QNN (Statevector + Reverse-Mode AD)
3. IBM Qiskit Hybrid QNN (Statevector + Analytical Parameter-Shift)

Dataset: Customer Reviews (CR) — Train: 3,024, Val: 362, Test: 384, Features: 4,760

Outputs:
- results/qiskit/three_way_comparison.csv
- results/qiskit/figures/framework_comparison_accuracy.png
- results/qiskit/figures/framework_comparison_f1.png
- results/qiskit/figures/framework_comparison_auc.png
- results/qiskit/figures/framework_comparison_time.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    os.makedirs("results/qiskit/figures", exist_ok=True)
    
    # 1. Load PennyLane & Classical Baseline Results from model_comparison.csv
    pl_path = "results/model_comparison.csv"
    if not os.path.exists(pl_path):
        raise FileNotFoundError(f"Missing {pl_path}")
    pl_df = pd.read_csv(pl_path)
    cr_pl = pl_df[pl_df["dataset"] == "CR"].copy()

    # 2. Load Genuine Qiskit Results from qiskit_model_results.csv
    qk_path = "results/qiskit/qiskit_model_results.csv"
    if not os.path.exists(qk_path):
        raise FileNotFoundError(f"Missing {qk_path}. Qiskit training must complete before benchmarking.")
    qk_df = pd.read_csv(qk_path)

    # 3. Assemble Unified Comparison Records
    rows = []

    # Map parameter counts and epochs for CR models
    param_map = {
        "Logistic Regression": 4761,
        "Linear SVM": 4761,
        "Classical MLP": 306850,
        "Compact MLP (4-dim)": 19054,
        "Hybrid QNN": 19078
    }
    epochs_map = {
        "Logistic Regression": "N/A (Convex)",
        "Linear SVM": "N/A (Convex)",
        "Classical MLP": 10,
        "Compact MLP (4-dim)": 10,
        "Hybrid QNN": 10
    }

    for _, r in cr_pl.iterrows():
        m_name = r["model"]
        fw = "PennyLane" if "Hybrid QNN" in m_name else "Classical ML"
        # Positive class F1 and Macro F1 calculation from TP, TN, FP, FN
        tp, tn, fp, fn = r["tp"], r["tn"], r["fp"], r["fn"]
        pos_p = tp / (tp + fp) if (tp + fp) > 0 else 0
        pos_r = tp / (tp + fn) if (tp + fn) > 0 else 0
        pos_f1 = (2 * pos_p * pos_r / (pos_p + pos_r)) if (pos_p + pos_r) > 0 else 0
        neg_p = tn / (tn + fn) if (tn + fn) > 0 else 0
        neg_r = tn / (tn + fp) if (tn + fp) > 0 else 0
        neg_f1 = (2 * neg_p * neg_r / (neg_p + neg_r)) if (neg_p + neg_r) > 0 else 0
        macro_f1 = (pos_f1 + neg_f1) / 2.0

        rows.append({
            "dataset": "CR",
            "framework": fw,
            "model": m_name,
            "evaluation_split": "test (384 samples)",
            "accuracy": r["accuracy"] * 100.0,
            "weighted_precision": r["precision"] * 100.0,
            "weighted_recall": r["recall"] * 100.0,
            "weighted_f1": r["f1_score"] * 100.0,
            "macro_f1": macro_f1 * 100.0,
            "pos_f1": pos_f1 * 100.0,
            "auc_roc": r["auc_roc"] * 100.0,
            "total_train_time_sec": r["train_time"],
            "per_epoch_time_sec": r["train_time"] / 10.0 if isinstance(epochs_map[m_name], int) else np.nan,
            "eval_time_sec": r["eval_time"],
            "total_epochs": epochs_map[m_name],
            "trainable_parameters": param_map[m_name],
            "random_seed": 42,
            "run_identifier": "seed_42_cr_benchmark"
        })

    # Add Qiskit Row
    for _, r in qk_df.iterrows():
        rows.append({
            "dataset": "CR",
            "framework": "IBM Qiskit",
            "model": "Hybrid QNN (IBM Qiskit)",
            "evaluation_split": "test (384 samples)",
            "accuracy": r["accuracy"] * 100.0,
            "weighted_precision": r["precision"] * 100.0,
            "weighted_recall": r["recall"] * 100.0,
            "weighted_f1": r["weighted_f1"] * 100.0,
            "macro_f1": r["macro_f1"] * 100.0,
            "pos_f1": r["pos_f1"] * 100.0,
            "auc_roc": r["auc_roc"] * 100.0,
            "total_train_time_sec": r["train_time_sec"],
            "per_epoch_time_sec": r["train_time_sec"] / r["total_epochs"],
            "eval_time_sec": r["eval_time_sec"],
            "total_epochs": int(r["total_epochs"]),
            "trainable_parameters": int(r["trainable_parameters"]),
            "random_seed": int(r["seed"]),
            "run_identifier": "seed_42_cr_qiskit_run"
        })

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv("results/qiskit/three_way_comparison.csv", index=False)
    print("Saved results/qiskit/three_way_comparison.csv\n")
    print(comp_df[["framework", "model", "accuracy", "weighted_f1", "auc_roc", "total_train_time_sec"]].to_string(index=False))

    # Aesthetics
    sns.set_theme(style="whitegrid", palette="muted")
    colors = {
        "Classical ML": "#4A90E2",
        "PennyLane": "#50E3C2",
        "IBM Qiskit": "#9013FE"
    }
    bar_colors = [colors[f] for f in comp_df["framework"]]

    # 1. Accuracy Comparison Chart
    plt.figure(figsize=(9, 5))
    bars = plt.bar(comp_df["model"], comp_df["accuracy"], color=bar_colors, edgecolor="black", alpha=0.85)
    plt.title("Customer Reviews (CR): Verified Test Accuracy Across Frameworks", fontsize=12, fontweight="bold")
    plt.ylabel("Test Accuracy (%)", fontsize=11)
    plt.ylim(70, 85)
    plt.xticks(rotation=20, ha="right", fontsize=10)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, f"{yval:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/qiskit/figures/framework_comparison_accuracy.png", dpi=300)
    plt.close()
    print("Saved framework_comparison_accuracy.png")

    # 2. Weighted F1 Comparison Chart
    plt.figure(figsize=(9, 5))
    bars = plt.bar(comp_df["model"], comp_df["weighted_f1"], color=bar_colors, edgecolor="black", alpha=0.85)
    plt.title("Customer Reviews (CR): Verified Weighted F1-Score Across Frameworks", fontsize=12, fontweight="bold")
    plt.ylabel("Weighted F1-Score (%)", fontsize=11)
    plt.ylim(70, 85)
    plt.xticks(rotation=20, ha="right", fontsize=10)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, f"{yval:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/qiskit/figures/framework_comparison_f1.png", dpi=300)
    plt.close()
    print("Saved framework_comparison_f1.png")

    # 3. AUC-ROC Comparison Chart
    plt.figure(figsize=(9, 5))
    bars = plt.bar(comp_df["model"], comp_df["auc_roc"], color=bar_colors, edgecolor="black", alpha=0.85)
    plt.title("Customer Reviews (CR): Verified ROC-AUC Across Frameworks", fontsize=12, fontweight="bold")
    plt.ylabel("ROC-AUC (%)", fontsize=11)
    plt.ylim(80, 92)
    plt.xticks(rotation=20, ha="right", fontsize=10)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, f"{yval:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/qiskit/figures/framework_comparison_auc.png", dpi=300)
    plt.close()
    print("Saved framework_comparison_auc.png")

    # 4. Training Time Comparison (Log Scale)
    plt.figure(figsize=(9, 5))
    bars = plt.bar(comp_df["model"], comp_df["total_train_time_sec"], color=bar_colors, edgecolor="black", alpha=0.85)
    plt.title("Customer Reviews (CR): Wall-Clock Training Duration (Log Scale)", fontsize=12, fontweight="bold")
    plt.ylabel("Total Training Duration (seconds, log scale)", fontsize=11)
    plt.yscale("log")
    plt.xticks(rotation=20, ha="right", fontsize=10)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval * 1.15, f"{yval:.2f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/qiskit/figures/framework_comparison_time.png", dpi=300)
    plt.close()
    print("Saved framework_comparison_time.png")

if __name__ == "__main__":
    main()
