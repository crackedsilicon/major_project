"""Run a short class-incremental federated-learning demonstration."""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='synthetic', help='Processed dataset folder name')
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--exp', default='cil_demo')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_experiment(
        dataset=args.dataset,
        rounds=args.rounds,
        epochs=args.epochs,
        exp_name=args.exp,
        device='cpu',
        agg_strategy='adaptive',
        seed=args.seed,
        class_stages=[[0], [1]],
    )


if __name__ == '__main__':
    main()