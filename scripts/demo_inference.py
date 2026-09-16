#!/usr/bin/env python
"""
demo_inference.py

Offline "IDS alert" demonstration wrapper for the Adaptive Federated
Class-Incremental IoT Intrusion Detection project.

WHAT THIS DOES
---------------
Loads a trained GLOBAL model checkpoint (produced by aggregating federated clients),
plus a set of traffic records (curated or client sample), runs them through the
model one at a time, and prints a clean NORMAL / IDS ALERT verdict for each record.

USAGE
-----
    python scripts/demo_inference.py --dataset synthetic --exp cil_demo --delay 0.5
    python scripts/demo_inference.py --dataset synthetic --client 0 --num-samples 12 --delay 1.0
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("PyTorch is required. Activate your project environment first:")
    print(r"    .\.venv\Scripts\Activate.ps1")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 1. MODEL IMPORT
# ---------------------------------------------------------------------------
try:
    from models.simple_nn import SimpleNet as ModelClass
    USING_PLACEHOLDER_MODEL = False
except ImportError:
    USING_PLACEHOLDER_MODEL = True

    class ModelClass(nn.Module):
        """Placeholder architecture if project import fails."""

        def __init__(self, input_dim=20, num_classes=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes),
            )

        def forward(self, x):
            return self.net(x)


# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def load_class_names(meta_path, num_classes):
    """Best-effort load of human-readable class names from meta.npz."""
    names = [f"Class {i}" for i in range(num_classes)]
    if not os.path.exists(meta_path):
        return names
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


def guess_normal_class(class_names, override=None):
    if override is not None:
        return override
    for i, name in enumerate(class_names):
        if "normal" in name.lower() or "benign" in name.lower():
            return i
    return 0  # default assumption


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"{YELLOW}No checkpoint found at {checkpoint_path}.{RESET}")
        print("Ensure runner.py saves the checkpoint upon experiment completion:")
        print(f'    torch.save(global_model.state_dict(), "{checkpoint_path}")')
        sys.exit(1)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"{YELLOW}Strict state_dict load failed, trying non-strict: {e}{RESET}")
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def pick_demo_records(X, y, normal_class, num_samples, seed):
    rng = np.random.default_rng(seed)
    normal_idx = np.where(y == normal_class)[0]
    attack_idx = np.where(y != normal_class)[0]
    n_normal = min(num_samples // 2, len(normal_idx))
    n_attack = min(num_samples - n_normal, len(attack_idx))
    parts = []
    if n_normal:
        parts.append(rng.choice(normal_idx, size=n_normal, replace=False))
    if n_attack:
        parts.append(rng.choice(attack_idx, size=n_attack, replace=False))
    chosen = np.concatenate(parts).astype(int) if parts else np.array([], dtype=int)
    rng.shuffle(chosen)
    return X[chosen], y[chosen]


# ---------------------------------------------------------------------------
# 3. Main demo loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Offline IDS alert demo")
    parser.add_argument("--dataset", default="synthetic",
                        help="Folder name under data/processed/ (e.g. synthetic or 'TON_IoT datasets')")
    parser.add_argument("--client", type=int, default=0,
                        help="Which client_N.npz file to draw demo records from")
    parser.add_argument("--records-path", default=None,
                        help="Optional explicit path to curated records .npz file (e.g. data/demo_records.npz)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to saved global model .pt file (defaults to results/<exp>/global_model.pt)")
    parser.add_argument("--exp", default="cil_demo",
                        help="Experiment/results folder name, used for default checkpoint path")
    parser.add_argument("--num-samples", type=int, default=12,
                        help="How many demo records to run through the model")
    parser.add_argument("--normal-class", type=int, default=None,
                        help="Override which integer label counts as 'normal'")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds to pause between each record, for demo pacing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = os.path.join("data", "processed", args.dataset)
    meta_path = os.path.join(data_dir, "meta.npz")
    checkpoint_path = args.checkpoint or os.path.join("results", args.exp, "global_model.pt")

    # Load records source
    if args.records_path and os.path.exists(args.records_path):
        data = np.load(args.records_path, allow_pickle=True)
        Xd, yd = data["X"], data["y"]
        if "class_names" in data:
            class_names = [str(n) for n in data["class_names"]]
            num_classes = len(class_names)
        else:
            num_classes = int(yd.max()) + 1
            class_names = load_class_names(meta_path, num_classes)
        print(f"Loaded {len(Xd)} records from curated file {args.records_path}")
    else:
        client_path = os.path.join(data_dir, "clients", f"client_{args.client}.npz")
        if not os.path.exists(client_path):
            print(f"Could not find {client_path}. Check --dataset / --client.")
            sys.exit(1)
        data = np.load(client_path)
        X, y = data["X"], data["y"]
        num_classes = int(y.max()) + 1
        class_names = load_class_names(meta_path, num_classes)
        normal_class = guess_normal_class(class_names, args.normal_class)
        Xd, yd = pick_demo_records(X, y, normal_class, args.num_samples, args.seed)

    normal_class = guess_normal_class(class_names, args.normal_class)

    if len(Xd) == 0:
        print("No demo records could be selected -- check dataset/client path.")
        sys.exit(1)

    model = ModelClass(input_dim=Xd.shape[1], num_classes=num_classes)
    if USING_PLACEHOLDER_MODEL:
        print(f"{YELLOW}Warning: using placeholder architecture -- check model import.{RESET}\n")

    model = load_checkpoint(model, checkpoint_path)

    print(f"\n{BOLD}===================================================={RESET}")
    print(f"{BOLD}   Adaptive Federated IoT IDS -- Live Alert Demo   {RESET}")
    print(f"{BOLD}===================================================={RESET}")
    print(f"Dataset: {args.dataset} | Experiment: {args.exp} | Checkpoint: {checkpoint_path}")
    print(f"Evaluating {len(Xd)} streaming traffic flows...\n")

    correct = 0
    with torch.no_grad():
        for i in range(len(Xd)):
            x = torch.tensor(Xd[i:i + 1], dtype=torch.float32)
            logits = model(x)
            pred = int(logits.argmax(dim=1).item())
            actual = int(yd[i])
            is_correct = (pred == actual)
            correct += int(is_correct)

            time.sleep(args.delay)
            actual_label = class_names[actual] if actual < len(class_names) else f"Class {actual}"
            pred_label = class_names[pred] if pred < len(class_names) else f"Class {pred}"

            if pred == normal_class:
                verdict = f"{GREEN}{BOLD}[NORMAL TRAFFIC]{RESET}"
            else:
                verdict = f"{RED}{BOLD}[IDS ALERT: {pred_label.upper()} DETECTED]{RESET}"

            mark = f"{GREEN}OK{RESET}" if is_correct else f"{YELLOW}MISCLASSIFIED{RESET}"
            print(f"[{i + 1:02d}/{len(Xd):02d}] Flow Target: {actual_label:<15} -> Verdict: {verdict:<45} ({mark})")

    acc = correct / len(Xd)
    print(f"\n{BOLD}----------------------------------------------------{RESET}")
    print(f"{BOLD}Demo Batch Inference Accuracy: {acc:.2%} ({correct}/{len(Xd)}){RESET}")
    print(f"{BOLD}----------------------------------------------------{RESET}\n")


if __name__ == "__main__":
    main()
