# Dataset Dispatcher

from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split

# CR
def _load_cr(seed=42):
    file_path = "data/cr/custrev.txt"

    texts, labels = [], []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label, sentence = line.split("\t", 1)
            labels.append(int(label))
            texts.append(sentence)

    texts = np.array(texts)
    labels = np.array(labels)

    train_size = 3024
    test_size = 384

    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, train_size=train_size,
        stratify=labels, random_state=seed
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size,
        stratify=y_temp, random_state=seed
    )

    print("\n[CR Split]")
    print(f"Train: {len(X_train):,}")
    print(f"Val  : {len(X_val):,}")
    print(f"Test : {len(X_test):,}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# MPQA
def _load_mpqa(seed=42):

    def sanitize(x):
        return [str(s).strip() for s in x]

    def normalize_labels(y):
        y = np.asarray(y)
        uniq = set(np.unique(y).tolist())
        if uniq == {-1, 1}:
            return (y == 1).astype(np.int64)
        if uniq == {0, 1}:
            return y.astype(np.int64)
        if uniq == {1, 2}:
            return (y == 2).astype(np.int64)
        lo, hi = sorted(list(uniq))
        return (y == hi).astype(np.int64)

    ds = load_dataset("jxm/mpqa") # Using Public Path

    texts = (
        sanitize(ds["train"]["sentence"]) +
        sanitize(ds["dev"]["sentence"]) +
        sanitize(ds["test"]["sentence"])
    )

    labels = np.concatenate([
        normalize_labels(ds["train"]["label"]),
        normalize_labels(ds["dev"]["label"]),
        normalize_labels(ds["test"]["label"])
    ])

    total = len(texts)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)

    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, train_size=train_size,
        stratify=labels, random_state=seed
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=len(texts) - train_size - val_size,
        stratify=y_temp, random_state=seed
    )

    print("\n[MPQA Split - 8:1:1]")
    print(f"Train: {len(X_train):,}")
    print(f"Val  : {len(X_val):,}")
    print(f"Test : {len(X_test):,}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# MR
def _load_mr(seed=42):
    base = "data/mr"

    train_text_path  = f"{base}/text_train.txt"
    test_text_path   = f"{base}/text_test.txt"
    train_label_path = f"{base}/label_train.txt"
    test_label_path  = f"{base}/label_test.txt"

    def load_lines(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return [line.strip() for line in f if line.strip()]

    X_train_text = load_lines(train_text_path)
    X_test_text  = load_lines(test_text_path)

    y_train = np.array([int(line.strip()) for line in open(train_label_path) if line.strip()])
    y_test  = np.array([int(line.strip()) for line in open(test_label_path)  if line.strip()])

    X_all = X_train_text + X_test_text
    y_all = np.concatenate([y_train, y_test])

    train_size = 8530
    val_size   = 1065
    test_size  = 1067

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, train_size=train_size,
        stratify=y_all, random_state=seed
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size,
        stratify=y_temp, random_state=seed
    )

    print("\n[MR Split]")
    print(f"Train: {len(X_train):,}")
    print(f"Val  : {len(X_val):,}")
    print(f"Test : {len(X_test):,}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# SUBJ
def _load_subj(seed=42):
    ds = load_dataset("SetFit/subj") # Using Public Path

    split_test = ds["test"].train_test_split(test_size=0.5, seed=seed)

    ds["validation"] = split_test["train"]
    ds["test"]       = split_test["test"]

    X_train = [x["text"] for x in ds["train"]]
    y_train = np.array([x["label"] for x in ds["train"]])

    X_val = [x["text"] for x in ds["validation"]]
    y_val = np.array([x["label"] for x in ds["validation"]])

    X_test = [x["text"] for x in ds["test"]]
    y_test = np.array([x["label"] for x in ds["test"]])

    print("\n[SUBJ Split]")
    print(f"Train: {len(X_train):,}")
    print(f"Val  : {len(X_val):,}")
    print(f"Test : {len(X_test):,}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# SST
def _load_sst(seed=42):
    base_dir = "data/sst"

    def load_list(path):
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    X_train = load_list(f"{base_dir}/text_train.txt")
    y_train = np.array([int(x) for x in load_list(f"{base_dir}/label_train.txt")])

    X_val = load_list(f"{base_dir}/text_val.txt")
    y_val = np.array([int(x) for x in load_list(f"{base_dir}/label_val.txt")])

    X_test = load_list(f"{base_dir}/text_test.txt")
    y_test = np.array([int(x) for x in load_list(f"{base_dir}/label_test.txt")])

    print("\n[SST Split]")
    print(f"Train: {len(X_train):,}")
    print(f"Val  : {len(X_val):,}")
    print(f"Test : {len(X_test):,}")

    return X_train, X_val, X_test, y_train, y_val, y_test

# Dispatcher
def load_text_dataset(name, seed=42):
    name = name.lower().strip()

    if name == "cr":
        return _load_cr(seed)
    elif name == "mpqa":
        return _load_mpqa(seed)
    elif name == "mr":
        return _load_mr(seed)
    elif name == "subj":
        return _load_subj(seed)
    elif name == "sst":
        return _load_sst(seed)
    else:
        raise ValueError(
            f"Unsupported dataset: {name}. Supported: 'cr', 'mpqa', 'mr', 'subj', 'sst'"
        )
