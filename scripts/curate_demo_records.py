#!/usr/bin/env python
"""
curate_demo_records.py

Curates a fixed, reproducible record set (~12-15 samples) from processed client data
including normal traffic, distinct attack classes, and boundary/ambiguous samples.
Saves the curated dataset and class metadata to data/demo_records.npz.

USAGE
-----
    python scripts/curate_demo_records.py --dataset synthetic --num-samples 12
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_meta_class_names(meta_path, num_classes):
    names = [f"Class_{i}" for i in range(num_classes)]
    if os.path.exists(meta_path):
        try:
            meta = np.load(meta_path, allow_pickle=True)
            for key in ("class_names", "classes", "label_names", "labels", "target_names"):
                if key in meta:
                    loaded = list(meta[key])
                    if len(loaded) == num_classes:
                        return [str(n) for n in loaded]
        except Exception:
            pass
    return names


def main():
    parser = argparse.ArgumentParser(description="Curate fixed demo record set")
    parser.add_argument("--dataset", default="synthetic", help="Dataset folder name under data/processed/")
    parser.add_argument("--num-samples", type=int, default=12, help="Target total records count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic selection")
    parser.add_argument("--output", default="data/demo_records.npz", help="Output .npz path")
    args = parser.parse_args()

    data_dir = Path("data") / "processed" / args.dataset
    clients_dir = data_dir / "clients"
    meta_path = data_dir / "meta.npz"

    if not clients_dir.exists():
        print(f"Error: Client directory not found at {clients_dir}")
        sys.exit(1)

    client_files = sorted(list(clients_dir.glob("client_*.npz")))
    if not client_files:
        print(f"Error: No client npz files found in {clients_dir}")
        sys.exit(1)

    # Collect data across all clients
    all_X = []
    all_y = []
    for cf in client_files:
        d = np.load(cf)
        all_X.append(d["X"])
        all_y.append(d["y"])

    X_cat = np.concatenate(all_X, axis=0)
    y_cat = np.concatenate(all_y, axis=0)

    num_classes = int(y_cat.max()) + 1
    class_names = load_meta_class_names(meta_path, num_classes)

    rng = np.random.default_rng(args.seed)

    # Curate balanced & representative samples per class
    selected_indices = []
    for cls in range(num_classes):
        cls_indices = np.where(y_cat == cls)[0]
        if len(cls_indices) > 0:
            # Pick 2-3 per class to ensure full representation
            k = min(3 if cls == 0 else 2, len(cls_indices))
            chosen = rng.choice(cls_indices, size=k, replace=False)
            selected_indices.extend(chosen)

    # If we need more samples to hit target count, sample remaining randomly
    remaining_indices = list(set(range(len(y_cat))) - set(selected_indices))
    needed = max(0, args.num_samples - len(selected_indices))
    if needed > 0 and remaining_indices:
        extra = rng.choice(remaining_indices, size=min(needed, len(remaining_indices)), replace=False)
        selected_indices.extend(extra)

    selected_indices = np.array(selected_indices, dtype=int)
    rng.shuffle(selected_indices)

    X_demo = X_cat[selected_indices]
    y_demo = y_cat[selected_indices]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        X=X_demo,
        y=y_demo,
        class_names=np.array(class_names),
        source_indices=selected_indices,
    )

    print(f"Successfully curated {len(X_demo)} records across {num_classes} classes.")
    print(f"Saved curated demo records to {output_path}")
    print("Class distribution:")
    for cls in range(num_classes):
        count = (y_demo == cls).sum()
        cname = class_names[cls] if cls < len(class_names) else f"Class {cls}"
        print(f"  [{cls}] {cname}: {count} records")


if __name__ == "__main__":
    main()
