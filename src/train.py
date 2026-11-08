import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import numpy as np
import argparse
import os

from src.model import HybridClassifier
from src.data import load_text_dataset


def train(cfg):
    # Data Loading
    X_train, X_val, X_test, y_train, y_val, y_test = load_text_dataset(cfg["dataset"])

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=cfg["max_features"])
    X_train_tfidf = vectorizer.fit_transform(list(X_train)).toarray()
    X_val_tfidf   = vectorizer.transform(list(X_val)).toarray()
    X_test_tfidf  = vectorizer.transform(list(X_test)).toarray()

    # Convert to Tensors
    X_train_tensor = torch.tensor(X_train_tfidf, dtype=torch.float32)
    X_val_tensor   = torch.tensor(X_val_tfidf,   dtype=torch.float32)
    X_test_tensor  = torch.tensor(X_test_tfidf,  dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor   = torch.tensor(y_val,   dtype=torch.long)
    y_test_tensor  = torch.tensor(y_test,  dtype=torch.long)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train_tensor = X_train_tensor.to(device)
    X_val_tensor   = X_val_tensor.to(device)
    X_test_tensor  = X_test_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    y_val_tensor   = y_val_tensor.to(device)
    y_test_tensor  = y_test_tensor.to(device)

    # Model
    model = HybridClassifier(
        input_dim=X_train_tfidf.shape[1],
        n_qubits=cfg["n_qubits"],
        n_layers=cfg["n_layers"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    best_val_f1 = 0.0
    best_epoch  = 0
    best_state  = None

    for epoch in range(cfg["n_epochs"]):
        model.train()
        permutation = torch.randperm(X_train_tensor.size(0))

        for i in range(0, X_train_tensor.size(0), cfg["batch_size"]):
            indices = permutation[i:i + cfg["batch_size"]]
            batch_x = X_train_tensor[indices]
            batch_y = y_train_tensor[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_pred    = torch.argmax(val_outputs, dim=1)
            val_acc = accuracy_score(y_val_tensor.cpu(), val_pred.cpu())
            val_f1  = f1_score(y_val_tensor.cpu(), val_pred.cpu(), average="weighted")

        print(f"Epoch {epoch + 1}/{cfg['n_epochs']} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch + 1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nBest checkpoint: epoch {best_epoch} (Val F1: {best_val_f1:.4f})")

    # Save Best Checkpoint
    os.makedirs("results", exist_ok=True)
    ckpt_path = f"results/{cfg['dataset']}_best.pt"
    torch.save(best_state, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    # Evaluate on Test Set
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_pred    = torch.argmax(test_outputs, dim=1)
        test_probs   = torch.softmax(test_outputs, dim=1)[:, 1]

        test_acc = accuracy_score(y_test_tensor.cpu(), test_pred.cpu())
        test_f1  = f1_score(y_test_tensor.cpu(), test_pred.cpu(), average="weighted")
        test_auc = roc_auc_score(y_test_tensor.cpu(), test_probs.cpu())

    print("\nFinal Test Results")
    print(f"Dataset  : {cfg['dataset']}")
    print(f"Accuracy : {test_acc:.4f}")
    print(f"F1-Score : {test_f1:.4f}")
    print(f"AUC-ROC  : {test_auc:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test_tensor.cpu(), test_pred.cpu(), digits=4))

    return {"accuracy": test_acc, "f1": test_f1, "auc": test_auc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      type=str,   default="MR")
    parser.add_argument("--n_qubits",     type=int,   default=4)
    parser.add_argument("--n_layers",     type=int,   default=2)
    parser.add_argument("--max_features", type=int,   default=5000)
    parser.add_argument("--lr",           type=float, default=0.001)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--n_epochs",     type=int,   default=10)
    args = parser.parse_args()

    cfg = vars(args)
    train(cfg)
