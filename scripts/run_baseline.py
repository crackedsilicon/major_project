"""Run a minimal federated baseline simulation using synthetic data.

This script is a small runnable baseline to verify the training, client,
and aggregation flow. Replace synthetic data with dataset loaders later.
"""
import argparse
import random
import sys
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.simple_nn import SimpleNet
from clients.client import Client
from server.aggregator import aggregate


def make_synthetic_dataset(n_samples=200, input_dim=20, n_classes=2, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(n_samples, input_dim)
    y = torch.randint(0, n_classes, (n_samples,))
    return TensorDataset(x, y)


def run_baseline(rounds=3, n_clients=3, epochs=1):
    input_dim = 20
    num_classes = 2
    global_model = SimpleNet(input_dim=input_dim, num_classes=num_classes)

    # create clients with small synthetic data
    clients = []
    client_loaders = {}
    for i in range(n_clients):
        ds = make_synthetic_dataset(n_samples=200, input_dim=input_dim, n_classes=num_classes, seed=42 + i)
        loader = DataLoader(ds, batch_size=32, shuffle=True)
        client_loaders[i] = loader
        clients.append(Client(client_id=i, model=global_model))

    # federated rounds
    for r in range(rounds):
        print(f"--- Round {r+1}/{rounds} ---")
        models = []
        metadatas = []

        for i, c in enumerate(clients):
            state, meta = c.local_train(client_loaders[i], epochs=epochs)
            models.append(state)
            # ensure metadata has data_size
            meta.setdefault('data_size', len(client_loaders[i].dataset))
            metadatas.append(meta)
            print(f"Client {i} local acc: {meta.get('accuracy'):.3f}")

        # aggregate
        agg_state = aggregate(models, metadatas)

        # load aggregated params into global model (best-effort)
        try:
            global_model.load_state_dict(agg_state, strict=False)
        except Exception as e:
            print("Warning: could not directly load aggregated state_dict:", e)

    # final evaluation on a held-out synthetic set
    test_ds = make_synthetic_dataset(n_samples=500, input_dim=input_dim, n_classes=num_classes, seed=999)
    test_loader = DataLoader(test_ds, batch_size=64)
    global_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            out = global_model(x)
            preds = out.argmax(dim=1)
            total += y.size(0)
            correct += (preds == y).sum().item()
    print(f"Final global accuracy (synthetic): {correct/total:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--clients', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=1)
    args = parser.parse_args()
    run_baseline(rounds=args.rounds, n_clients=args.clients, epochs=args.epochs)
