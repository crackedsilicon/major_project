"""Generate synthetic processed per-client .npz files for quick experiments."""
import numpy as np
from pathlib import Path
import argparse


def generate(dataset='synthetic', out='data/processed', n_clients=5, n_samples=200, input_dim=20, n_classes=2, seed=0):
    np.random.seed(seed)
    out_root = Path(out) / dataset
    clients_dir = out_root / 'clients'
    clients_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_clients):
        # create class skew for heterogeneity: each client favors a subset of classes
        probs = np.ones(n_classes)
        favored = i % n_classes
        probs[favored] = 3.0
        probs = probs / probs.sum()
        y = np.random.choice(np.arange(n_classes), size=n_samples, p=probs)
        X = np.random.randn(n_samples, input_dim)
        np.savez_compressed(clients_dir / f'client_{i}.npz', X=X.astype(np.float32), y=y.astype(np.int64))
        print(f"Wrote client_{i}.npz with {n_samples} samples")

    np.savez_compressed(out_root / 'meta.npz', label_col='label')
    print(f"Synthetic processed dataset written to {out_root}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='synthetic')
    parser.add_argument('--out', type=str, default='data/processed')
    parser.add_argument('--n_clients', type=int, default=5)
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--input_dim', type=int, default=20)
    parser.add_argument('--n_classes', type=int, default=2)
    args = parser.parse_args()
    generate(dataset=args.dataset, out=args.out, n_clients=args.n_clients, n_samples=args.n_samples, input_dim=args.input_dim, n_classes=args.n_classes)
