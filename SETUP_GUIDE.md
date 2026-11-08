# Setup Guide
## Quantum Neural Network for Text Classification

---

## System Requirements

| Component | Minimum | Tested On |
|-----------|---------|-----------|
| OS | macOS 13+ / Linux | macOS 15.5 (Sequoia) |
| Architecture | x86_64 / arm64 | Apple Silicon (arm64) |
| Python | 3.10+ | 3.14.6 |
| RAM | 4 GB | 8 GB |
| Storage | 2 GB free | — |

> **Note:** This project runs entirely on CPU. No GPU or CUDA required.

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/AnnyshaHuzzat/Primer_QNN_Text_Classification.git
cd Primer_QNN_Text_Classification
```

---

## Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
```

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Verified Package Versions

| Package | Version |
|---------|---------|
| pennylane | 0.45.1 |
| torch | 2.15.0 |
| scikit-learn | 1.9.1 |
| numpy | 2.5.4 |
| datasets | 5.1.0 |
| pandas | 3.1.0 |
| matplotlib | 3.11.2 |
| seaborn | 0.13.2 |
| scipy | 1.18.1 |
| tqdm | 4.70.1 |
| pyyaml | 6.0.3 |

---

## Step 4: Download Datasets

```bash
python download_datasets.py
```

This automatically downloads MR, CR, and SST datasets from Hugging Face and creates the required local file structure in `data/`. SUBJ and MPQA are downloaded automatically during training.

---

## Step 5: Verify Installation

```bash
python -c "
import pennylane as qml
import torch
print('PennyLane:', qml.__version__)
print('PyTorch:', torch.__version__)

# Quick quantum circuit test
dev = qml.device('default.qubit', wires=2)
@qml.qnode(dev, interface='torch')
def circuit(x):
    qml.RX(x[0], wires=0)
    qml.RY(x[1], wires=1)
    qml.CNOT(wires=[0,1])
    return qml.expval(qml.PauliZ(0))

x = torch.tensor([0.5, 0.3], requires_grad=True)
result = circuit(x)
result.backward()
print('Quantum circuit OK. Gradient:', x.grad)
"
```

---

## Step 6: Run Training

### Single Dataset (Original Script)
```bash
python -m src.train --dataset SUBJ
```

### Full Experiments (All Datasets + Baselines + Ablation)
```bash
python run_experiments.py
```

### Available Datasets
| Flag | Dataset | Source |
|------|---------|--------|
| `MR` | Movie Review | Local (data/mr/) |
| `CR` | Customer Review | Local (data/cr/) |
| `SST` | Stanford Sentiment | Local (data/sst/) |
| `SUBJ` | Subjectivity | Hugging Face (SetFit/subj) |
| `MPQA` | Opinion Polarity | Hugging Face (jxm/mpqa) |

### Training Parameters
| Parameter | Default | CLI Flag |
|-----------|---------|----------|
| Max TF-IDF features | 5000 | `--max_features` |
| Number of qubits | 4 | `--n_qubits` |
| Number of VQC layers | 2 | `--n_layers` |
| Learning rate | 0.001 | `--lr` |
| Batch size | 32 | `--batch_size` |
| Epochs | 10 | `--n_epochs` |

---

## Output Directory Structure

```
results/
├── model_comparison.csv          # All model results
├── dataset_statistics.csv        # Dataset info
├── ablation_study.csv            # Ablation experiments
├── experimental_summary.json     # Summary with best results
├── figures/                      # All visualizations
│   ├── comparison_*.png          # Bar charts
│   ├── *_cm.png                  # Confusion matrices
│   ├── *_roc.png                 # ROC curves
│   ├── *_curves.png              # Training curves
│   └── ablation_*.png            # Ablation plots
├── predictions/                  # Saved predictions
│   └── *_preds.npz               # y_true, y_pred, y_prob arrays
checkpoints/
├── *_QNN_best.pt                 # QNN model checkpoints
├── *_MLP_best.pt                 # MLP model checkpoints
logs/
└── experiment_*.log              # Full experiment logs
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: pennylane` | Activate venv: `source venv/bin/activate` |
| Dataset path errors | Run `python download_datasets.py` first |
| SUBJ/MPQA download fails | Check internet connection; uses Hugging Face |
| Out of memory | Reduce `--max_features` or `--batch_size` |
| Slow training | Expected on CPU; QNN ~2-5 min per epoch depending on dataset size |

