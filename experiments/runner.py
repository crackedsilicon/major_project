"""Experiment runner to simulate federated rounds.

Loads per-client processed `.npz` files from `data/processed/<dataset>/clients/`,
creates `Client` instances, runs local training rounds and aggregates using
`server.aggregator.aggregate`. Logs per-round metrics and estimated communication
overhead to `results/<exp_name>/`.
"""
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Sized, cast
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.simple_nn import SimpleNet
from clients.client import Client
from server.aggregator import aggregate, compute_weights


def load_clients_from_processed(dataset: str, device='cpu', seed=0):
    root = Path('data') / 'processed' / dataset / 'clients'
    if not root.exists():
        raise FileNotFoundError(f"Processed clients not found: {root}")
    files = sorted(list(root.glob('client_*.npz')))
    clients = []
    loaders = {}
    val_loaders = {}
    all_loaders = {}
    max_class = -1
    for i, f in enumerate(files):
        arr = np.load(str(f))
        X = torch.tensor(arr['X'], dtype=torch.float32)
        y = torch.tensor(arr['y'], dtype=torch.long)
        ds = TensorDataset(X, y)
        max_class = max(max_class, int(y.max().item()))
        split_generator = torch.Generator().manual_seed(seed + i)
        val_size = max(1, int(len(ds) * 0.2)) if len(ds) > 1 else 0
        train_size = len(ds) - val_size
        if val_size:
            train_ds, val_ds = torch.utils.data.random_split(
                ds, [train_size, val_size], generator=split_generator
            )
        else:
            train_ds, val_ds = ds, None
        loader = DataLoader(train_ds, batch_size=32, shuffle=True, generator=split_generator)
        all_loader = DataLoader(ds, batch_size=32, shuffle=False)
        loaders[i] = loader
        all_loaders[i] = all_loader
        if val_ds is not None:
            val_loaders[i] = DataLoader(val_ds, batch_size=32, shuffle=False)
        # instantiate client with a fresh model
        model = SimpleNet(input_dim=X.shape[1], num_classes=1)
        clients.append(Client(client_id=i, model=model, device=device))
    num_classes = max_class + 1
    for client in clients:
        client.model = SimpleNet(input_dim=client.model.net[0].in_features, num_classes=num_classes).to(device)
    return clients, loaders, val_loaders, all_loaders, num_classes


def estimate_bytes_state_dict(state_dict):
    total = 0
    for v in state_dict.values():
        if isinstance(v, torch.Tensor):
            total += v.numel() * 4  # assume float32
    return total


def filter_loader_by_classes(loader, class_ids: list[int]):
    """Build a loader containing only samples from the current CIL stage."""
    samples = []
    feature_shape = None
    for x, y in loader:
        feature_shape = x.shape[1:]
        mask = torch.zeros_like(y, dtype=torch.bool)
        for class_id in class_ids:
            mask |= y == class_id
        samples.extend(zip(x[mask], y[mask]))
    if feature_shape is None:
        raise ValueError("Cannot filter an empty source loader")
    if samples:
        xs = torch.stack([sample[0] for sample in samples])
        ys = torch.stack([sample[1] for sample in samples]).long()
    else:
        xs = torch.empty((0, *feature_shape), dtype=torch.float32)
        ys = torch.empty((0,), dtype=torch.long)
    return DataLoader(TensorDataset(xs, ys), batch_size=32, shuffle=False)


def run_experiment(dataset: str, rounds=5, epochs=1, exp_name='exp', device='cpu', compress=False, agg_strategy='adaptive', mu=0.0, seed=0, class_stages=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    clients, loaders, val_loaders, all_loaders, num_classes = load_clients_from_processed(dataset, device=device, seed=seed)
    n_clients = len(clients)
    print(f"Loaded {n_clients} clients for dataset {dataset}")

    # create a global model matching first client
    input_dim = next(iter(loaders.values())).dataset[0][0].shape[0]
    global_model = SimpleNet(input_dim=input_dim, num_classes=num_classes)

    results_dir = Path('results') / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)
    log = {'rounds': []}
    seen_classes = set()
    previous_class_accuracy = {}

    for r in range(rounds):
        print(f"--- Round {r+1}/{rounds} ---")
        stage_classes = None
        if class_stages:
            stage_classes = class_stages[min(r, len(class_stages) - 1)]
            print(f"CIL stage classes: {stage_classes}")
        old_classes = sorted(seen_classes)
        if stage_classes is not None:
            seen_classes.update(stage_classes)
        seen_class_list = sorted(seen_classes)
        models = []
        metadatas = []
        comm_bytes = 0

        # Each client receives the current global model as teacher for distillation
        teacher_state = global_model.state_dict()
        for i, c in enumerate(clients):
            train_loader = loaders[i]
            val_loader = val_loaders.get(i)
            if stage_classes is not None:
                train_loader = filter_loader_by_classes(train_loader, stage_classes)
                if val_loader is not None:
                    val_loader = filter_loader_by_classes(val_loader, stage_classes)
            # If FedProx is requested (agg_strategy == 'fedprox' and mu>0) pass global state and mu
            if agg_strategy == 'fedprox' and mu > 0.0:
                state, meta = c.local_train(train_loader, epochs=epochs, teacher_state_dict=teacher_state, distill_lambda=0.3, mu=mu, global_state=global_model.state_dict())
            else:
                state, meta = c.local_train(train_loader, epochs=epochs, teacher_state_dict=teacher_state, distill_lambda=0.3)

            # ensure tensors and convert to torch tensors for aggregation
            models.append({k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in state.items()})
            if val_loader is not None and len(cast(Sized, val_loader.dataset)) > 0:
                meta['val_score'] = c.evaluate(val_loader)['accuracy']
            else:
                meta['val_score'] = meta.get('accuracy', 0.0)
            meta['data_size'] = len(cast(Sized, train_loader.dataset))
            metadatas.append(meta)

            # communication estimate: either quantized bytes or float32 estimation
            if compress:
                from server.compression import quantize_state_dict_bytes
                comm_bytes += quantize_state_dict_bytes(state, dtype=torch.float16)
            else:
                comm_bytes += estimate_bytes_state_dict(state)

            print(f"Client {i} local acc: {meta.get('accuracy'):.3f}")

        agg_state = aggregate(models, metadatas, strategy=agg_strategy)
        aggregation_weights = compute_weights(metadatas, strategy=agg_strategy)
        try:
            global_model.load_state_dict(agg_state, strict=False)
        except Exception:
            pass

        # Collect prototypes from clients and aggregate into global prototypes
        client_prototypes = []
        proto_weights = []
        for i, c in enumerate(clients):
            prot = c.compute_prototypes(all_loaders[i])
            client_prototypes.append(prot)
            proto_weights.append(len(all_loaders[i].dataset))

        # aggregate prototypes per class weighted by client data_size
        agg_prototypes = {}
        proto_counts = {}
        for prot, w in zip(client_prototypes, proto_weights):
            for cls, vec in prot.items():
                if cls not in agg_prototypes:
                    agg_prototypes[cls] = vec.clone() * float(w)
                    proto_counts[cls] = float(w)
                else:
                    agg_prototypes[cls] += vec.clone() * float(w)
                    proto_counts[cls] += float(w)
        for cls in list(agg_prototypes.keys()):
            agg_prototypes[cls] = (agg_prototypes[cls] / proto_counts[cls]).clone()

        # prototype-based evaluation on concatenated client data
        proto_correct = 0
        proto_total = 0
        global_model.eval()
        with torch.no_grad():
            for loader in all_loaders.values():
                for x, y in loader:
                    # compute features with global_model
                    if hasattr(global_model, 'net') and isinstance(global_model.net, torch.nn.Sequential):
                        feats = x
                        for layer in list(global_model.net.children())[:-1]:
                            feats = layer(feats)
                    else:
                        feats = global_model(x)

                    for fi, yi in zip(feats, y):
                        # find nearest prototype
                        best_cls = None
                        best_dist = None
                        for cls, vec in agg_prototypes.items():
                            d = torch.norm(fi.cpu() - vec)
                            if best_dist is None or d < best_dist:
                                best_dist = d
                                best_cls = cls
                        proto_total += 1
                        if best_cls == int(yi.item()):
                            proto_correct += 1
        proto_acc = (proto_correct / proto_total) if proto_total > 0 else 0.0

        # Evaluate on all client data and retain class-level metrics for CIL analysis.
        correct = 0
        total = 0
        ys = []
        ps = []
        global_model.eval()
        with torch.no_grad():
            for loader in all_loaders.values():
                for x, y in loader:
                    out = global_model(x)
                    preds = out.argmax(dim=1)
                    total += y.size(0)
                    correct += (preds == y).sum().item()
                    ys.extend(y.tolist())
                    ps.extend(preds.tolist())
        global_acc = (correct / total) if total > 0 else 0.0

        # compute per-class precision/recall/f1
        try:
            from sklearn.metrics import precision_recall_fscore_support
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(ys, ps, average='macro', zero_division=0)
        except Exception:
            precision_macro = recall_macro = f1_macro = None

        class_accuracy = {}
        for class_id in sorted(set(ys)):
            class_total = sum(label == class_id for label in ys)
            class_correct = sum(label == prediction == class_id for label, prediction in zip(ys, ps))
            class_accuracy[str(class_id)] = class_correct / class_total if class_total else 0.0

        def accuracy_for_classes(class_ids):
            selected = [(label, prediction) for label, prediction in zip(ys, ps) if label in class_ids]
            if not selected:
                return None
            return sum(label == prediction for label, prediction in selected) / len(selected)

        stage_metrics = None
        if stage_classes is not None:
            old_accuracy = accuracy_for_classes(set(old_classes))
            new_accuracy = accuracy_for_classes(set(stage_classes))
            seen_accuracy = accuracy_for_classes(set(seen_class_list))
            old_forgetting = [
                previous_class_accuracy[str(class_id)] - class_accuracy.get(str(class_id), 0.0)
                for class_id in old_classes
                if str(class_id) in previous_class_accuracy
            ]
            stage_metrics = {
                'old_accuracy': old_accuracy,
                'new_accuracy': new_accuracy,
                'seen_accuracy': seen_accuracy,
                'forgetting': sum(old_forgetting) / len(old_forgetting) if old_forgetting else 0.0,
            }
        previous_class_accuracy = class_accuracy

        round_entry = {
            'round': r,
            'global_accuracy': global_acc,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'prototype_accuracy': proto_acc,
            'communication_bytes': comm_bytes,
            'aggregation_weights': aggregation_weights,
            'class_stage': stage_classes,
            'class_accuracy': class_accuracy,
            'stage_metrics': stage_metrics,
            'metadata_summary': { 'n_clients': n_clients }
        }
        log['rounds'].append(round_entry)
        print(f"Round {r} global acc: {global_acc:.4f}, comm_bytes: {comm_bytes}")

    # save global model checkpoint
    checkpoint_path = results_dir / 'global_model.pt'
    torch.save(global_model.state_dict(), checkpoint_path)
    print(f"Saved global model checkpoint to {checkpoint_path}")

    # save log
    with open(results_dir / 'log.json', 'w') as fh:
        json.dump(log, fh, indent=2)
    print(f"Saved experiment log to {results_dir / 'log.json'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--exp', type=str, default='exp')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--compress', action='store_true', help='Estimate communication with float16 quantization')
    parser.add_argument('--aggregator', type=str, choices=['adaptive','fedavg','weighted','fedprox'], default='adaptive')
    parser.add_argument('--mu', type=float, default=0.0, help='FedProx proximal coefficient (mu)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducible experiments')
    parser.add_argument('--stages', type=str, default='', help='Optional CIL stages, e.g. 0|1,2|3,4')
    args = parser.parse_args()
    class_stages = None
    if args.stages:
        class_stages = [[int(class_id) for class_id in stage.split(',') if class_id.strip()] for stage in args.stages.split('|')]
    run_experiment(dataset=args.dataset, rounds=args.rounds, epochs=args.epochs, exp_name=args.exp, device=args.device, compress=args.compress, agg_strategy=args.aggregator, mu=args.mu, seed=args.seed, class_stages=class_stages)
