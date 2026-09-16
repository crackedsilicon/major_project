"""Plot experiment logs saved as JSON into result folders for slides and analysis.

Generates:
  1. accuracy_over_rounds.png: Global accuracy curve over rounds comparing strategies.
  2. forgetting_bar_chart.png: CIL forgetting vs retained seen class accuracy across stages.
  3. aggregation_weights.png: Dynamic adaptive aggregation weights per client over rounds.
  4. f1_macro.png / communication_bytes.png: Performance and communication metrics.

Usage:
    python scripts/plot_results.py --exp results/cil_demo results/multi_seed_comparison_adaptive_seed0
"""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_log(path: Path):
    if (path / 'log.json').exists():
        return json.loads((path / 'log.json').read_text())
    elif path.name.endswith('.json') and path.exists():
        return json.loads(path.read_text())
    return None


def plot_accuracy_over_rounds(exp_paths, out_dir):
    plt.figure(figsize=(8, 5))
    has_data = False
    for p in exp_paths:
        p_path = Path(p)
        log = load_log(p_path)
        if log is None:
            continue
        rounds = log.get('rounds', [])
        if not rounds:
            continue
        vals = [r.get('global_accuracy', 0.0) for r in rounds]
        x = range(1, len(vals) + 1)
        plt.plot(x, vals, marker='o', linewidth=2, label=p_path.name)
        has_data = True

    if has_data:
        plt.xlabel('Federated Round', fontsize=12)
        plt.ylabel('Global Accuracy', fontsize=12)
        plt.title('Global Accuracy over FL Rounds', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        out_file = out_dir / 'accuracy_over_rounds.png'
        plt.savefig(out_file, dpi=300)
        print(f"Wrote {out_file}")
    plt.close()


def plot_forgetting_bar_chart(exp_paths, out_dir):
    plt.figure(figsize=(8, 5))
    names = []
    forgetting_vals = []
    seen_acc_vals = []

    for p in exp_paths:
        p_path = Path(p)
        log = load_log(p_path)
        if log is None:
            continue
        rounds = log.get('rounds', [])
        # Check last round stage_metrics
        for r in reversed(rounds):
            sm = r.get('stage_metrics')
            if sm is not None and isinstance(sm, dict):
                names.append(p_path.name)
                forgetting_vals.append(sm.get('forgetting', 0.0))
                seen_acc_vals.append(sm.get('seen_accuracy', 0.0))
                break

    if names:
        x = np.arange(len(names))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        rects1 = ax.bar(x - width/2, seen_acc_vals, width, label='Retained (Seen) Accuracy', color='#2ca02c')
        rects2 = ax.bar(x + width/2, forgetting_vals, width, label='Catastrophic Forgetting', color='#d62728')

        ax.set_ylabel('Score / Ratio', fontsize=12)
        ax.set_title('CIL Performance: Retained Accuracy vs Catastrophic Forgetting', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5, axis='y')
        plt.tight_layout()
        out_file = out_dir / 'forgetting_bar_chart.png'
        plt.savefig(out_file, dpi=300)
        print(f"Wrote {out_file}")
    plt.close()


def plot_aggregation_weights(exp_paths, out_dir):
    for p in exp_paths:
        p_path = Path(p)
        log = load_log(p_path)
        if log is None:
            continue
        rounds = log.get('rounds', [])
        weights_per_round = [r.get('aggregation_weights', []) for r in rounds if 'aggregation_weights' in r]
        if not weights_per_round or not weights_per_round[0]:
            continue

        n_clients = len(weights_per_round[0])
        n_rounds = len(weights_per_round)

        plt.figure(figsize=(8, 5))
        weights_arr = np.array(weights_per_round) # (n_rounds, n_clients)
        rounds_x = range(1, n_rounds + 1)

        for c in range(n_clients):
            plt.plot(rounds_x, weights_arr[:, c], marker='s', linewidth=2, label=f'Client {c}')

        plt.xlabel('Federated Round', fontsize=12)
        plt.ylabel('Aggregation Weight', fontsize=12)
        plt.title(f'Adaptive Aggregation Client Weights ({p_path.name})', fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        out_file = out_dir / 'aggregation_weights.png'
        plt.savefig(out_file, dpi=300)
        print(f"Wrote {out_file}")
        plt.close()
        break  # plot first valid experiment weights


def plot_experiments(paths, out_dir='results/plots'):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy_over_rounds(paths, out_dir)
    plot_forgetting_bar_chart(paths, out_dir)
    plot_aggregation_weights(paths, out_dir)

    for metric in ['global_accuracy', 'f1_macro', 'communication_bytes']:
        plt.figure(figsize=(8, 5))
        has_data = False
        for p in paths:
            log = load_log(Path(p))
            if log is None:
                continue
            vals = [r.get(metric) for r in log.get('rounds', []) if r.get(metric) is not None]
            if vals:
                plt.plot(range(1, len(vals)+1), vals, marker='o', label=Path(p).name)
                has_data = True
        if has_data:
            plt.xlabel('Round', fontsize=12)
            plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
            plt.title(metric.replace('_', ' ').title(), fontsize=13, fontweight='bold')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            out_file = out_dir / f'{metric}.png'
            plt.savefig(out_file, dpi=300)
            print(f"Wrote {out_file}")
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_paths', nargs='*')
    parser.add_argument('--out', type=str, default='results/plots')
    args = parser.parse_args()

    paths = args.exp_paths
    if not paths:
        # Default scan results directory if no explicit paths given
        results_path = Path('results')
        if results_path.exists():
            paths = [str(p) for p in results_path.glob('*') if p.is_dir() and (p / 'log.json').exists()]

    if paths:
        plot_experiments(paths, out_dir=args.out)
    else:
        print("No experiment paths found under results/")
