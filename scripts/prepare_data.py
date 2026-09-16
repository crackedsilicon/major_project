"""Dataset preparation script.

Reads raw dataset CSV files placed under `data/raw/<dataset>/` and produces
preprocessed per-client numpy archives under `out/<dataset>/clients/`.

Supports IID splitting and Dirichlet non-IID label partitioning.
"""
import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def _read_raw_csvs(raw_dir: Path):
    # find any csv files in directory
    files = list(raw_dir.glob('**/*.csv'))
    if len(files) == 0:
        return None
    # concatenate
    dfs = [pd.read_csv(str(p), low_memory=False) for p in files]
    df = pd.concat(dfs, ignore_index=True)
    return df


def _infer_label_column(df: pd.DataFrame):
    # common label column names
    for name in ['label', 'Label', 'attack', 'class', 'target']:
        if name in df.columns:
            return name
    # fallback: last column
    return df.columns[-1]


def partition_iid(X, y, n_clients):
    n = X.shape[0]
    idx = np.random.permutation(n)
    parts = np.array_split(idx, n_clients)
    return parts


def partition_dirichlet(y, n_clients, alpha=0.5):
    # Dirichlet label partitioning
    labels = np.unique(y)
    parts = [[] for _ in range(n_clients)]
    for lbl in labels:
        idx = np.where(y == lbl)[0]
        proportions = np.random.dirichlet(alpha * np.ones(n_clients))
        # split indices according to proportions
        counts = (proportions * len(idx)).astype(int)
        # adjust counts to match total
        while counts.sum() < len(idx):
            counts[np.argmax(proportions)] += 1
        start = 0
        for c_idx, cnt in enumerate(counts):
            if cnt > 0:
                parts[c_idx].extend(idx[start:start+cnt].tolist())
            start += cnt
    return [np.array(p, dtype=int) for p in parts]


def prepare(dataset, out_dir, n_clients=5, split='iid', alpha=0.5, seed=0):
    np.random.seed(seed)
    raw_dir = Path('data') / 'raw' / dataset
    if not raw_dir.exists():
        print(f"Raw data directory not found: {raw_dir}. Place CSV files there.")
        return

    df = _read_raw_csvs(raw_dir)
    if df is None:
        print(f"No CSV files found under {raw_dir}")
        return

    label_col = _infer_label_column(df)
    print(f"Using label column: {label_col}")

    # drop columns that are obviously non-feature (timestamps, IDs)
    drop_like = [c for c in df.columns if 'time' in c.lower() or 'id' == c.lower()]
    df = df.drop(columns=[c for c in drop_like if c in df.columns], errors='ignore')

    df = df[df[label_col].notna()].copy()
    y_raw = df[label_col]
    y, label_values = pd.factorize(y_raw)
    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).values
    # if X has zero numeric columns, try to convert categoricals
    if X.shape[1] == 0:
        X = pd.get_dummies(df.drop(columns=[label_col])).values

    # normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    out_root = Path(out_dir) / dataset
    clients_dir = out_root / 'clients'
    clients_dir.mkdir(parents=True, exist_ok=True)

    if split == 'iid':
        parts = partition_iid(X, y, n_clients)
    else:
        parts = partition_dirichlet(y, n_clients, alpha=alpha)

    for i, idx in enumerate(parts):
        Xi = X[idx]
        yi = y[idx]
        np.savez_compressed(clients_dir / f'client_{i}.npz', X=Xi, y=yi)
        print(f"Saved client {i} with {len(idx)} samples -> {clients_dir / f'client_{i}.npz'}")

    # save metadata
    np.savez_compressed(out_root / 'meta.npz', label_col=label_col, label_values=label_values.astype(str))
    print(f"Prepared dataset saved to {out_root}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--out', type=str, default='data/processed')
    parser.add_argument('--n_clients', type=int, default=5)
    parser.add_argument('--split', type=str, choices=['iid', 'dirichlet'], default='iid')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    prepare(args.dataset, args.out, n_clients=args.n_clients, split=args.split, alpha=args.alpha, seed=args.seed)
