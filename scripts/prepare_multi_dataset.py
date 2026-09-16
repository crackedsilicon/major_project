#!/usr/bin/env python
"""
prepare_multi_dataset.py

Combines client partitions from multiple real datasets (e.g. TON_IoT, UNSW_NB15, Bot_IoT)
into a unified multi-domain client directory under data/processed/combined_iot/clients/.

Each client represents an edge gateway deployed in a different IoT environment/domain.
When trained with runner.py, ALL clients update ONE SINGLE UNIFIED GLOBAL MODEL.

USAGE
-----
    python scripts/prepare_multi_dataset.py --datasets TON_IoT UNSW_NB15 Bot_IoT --out combined_iot
"""

import argparse
import os
from pathlib import Path
import numpy as np


def combine_datasets(dataset_names, out_name="combined_iot"):
    data_root = Path("data") / "processed"
    out_dir = data_root / out_name / "clients"
    out_dir.mkdir(parents=True, exist_ok=True)

    client_id = 0
    all_class_names = []
    max_features = 20

    print(f"Combining client partitions from datasets: {dataset_names}")

    for ds in dataset_names:
        ds_dir = data_root / ds / "clients"
        if not ds_dir.exists():
            print(f"Warning: Processed clients for '{ds}' not found at {ds_dir}. Skipping.")
            continue

        meta_path = data_root / ds / "meta.npz"
        ds_classes = []
        if meta_path.exists():
            m = np.load(meta_path, allow_pickle=True)
            if "label_values" in m:
                ds_classes = [f"{ds}_{c}" for c in m["label_values"]]
            elif "class_names" in m:
                ds_classes = [f"{ds}_{c}" for c in m["class_names"]]

        all_class_names.extend(ds_classes)

        client_files = sorted(list(ds_dir.glob("client_*.npz")))
        for cf in client_files:
            d = np.load(cf)
            X, y = d["X"], d["y"]

            # Standardize feature dimension to max_features
            if X.shape[1] < max_features:
                pad = np.zeros((X.shape[0], max_features - X.shape[1]), dtype=np.float32)
                X = np.hstack([X, pad])
            elif X.shape[1] > max_features:
                X = X[:, :max_features]

            out_file = out_dir / f"client_{client_id}.npz"
            np.savez_compressed(out_file, X=X, y=y)
            print(f"  [Client {client_id}] Assigned from {ds}/{cf.name} -> {len(X)} samples")
            client_id += 1

    # Save combined metadata
    meta_out = data_root / out_name / "meta.npz"
    np.savez_compressed(meta_out, class_names=np.array(all_class_names if all_class_names else ["Normal", "Attack"]))
    print(f"\nSuccessfully created joint multi-dataset environment at: data/processed/{out_name}/")
    print(f"Total clients created: {client_id}")
    print(f"Run training with: python experiments/runner.py --dataset {out_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple datasets into a single FL client directory")
    parser.add_argument("--datasets", nargs="+", default=["synthetic", "TON_IoT_small"], help="Dataset names under data/processed/")
    parser.add_argument("--out", default="combined_iot", help="Output combined dataset folder name")
    args = parser.parse_args()
    combine_datasets(args.datasets, args.out)
