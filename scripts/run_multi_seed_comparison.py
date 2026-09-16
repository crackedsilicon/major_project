"""Run identical federated experiments across strategies and random seeds."""
import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.runner import run_experiment


def run_comparison(dataset, rounds, epochs, seeds, output_dir, stages=None):
    strategies = {
        'adaptive': 0.0,
        'fedavg': 0.0,
        'weighted': 0.0,
        'fedprox': 0.1,
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary = {}

    for strategy, mu in strategies.items():
        metrics = {'accuracy': [], 'macro_f1': [], 'communication_bytes': []}
        for seed in seeds:
            experiment_name = f'{output_path.name}_{strategy}_seed{seed}'
            run_experiment(
                dataset=dataset,
                rounds=rounds,
                epochs=epochs,
                exp_name=experiment_name,
                device='cpu',
                agg_strategy=strategy,
                mu=mu,
                seed=seed,
                class_stages=stages,
            )
            log_path = Path('results') / experiment_name / 'log.json'
            entries = json.loads(log_path.read_text())['rounds']
            final = entries[-1]
            metrics['accuracy'].append(final['global_accuracy'])
            metrics['macro_f1'].append(final['f1_macro'])
            metrics['communication_bytes'].append(final['communication_bytes'])
        summary[strategy] = {
            metric: {'mean': mean(values), 'std': pstdev(values) if len(values) > 1 else 0.0, 'values': values}
            for metric, values in metrics.items()
        }

    (output_path / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Saved summary to {output_path / "summary.json"}')
    for strategy, metrics in summary.items():
        print(
            f'{strategy}: accuracy={metrics["accuracy"]["mean"]:.4f} +/- {metrics["accuracy"]["std"]:.4f}, '
            f'macro_f1={metrics["macro_f1"]["mean"]:.4f} +/- {metrics["macro_f1"]["std"]:.4f}'
        )
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='synthetic')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--output-dir', default='results/multi_seed_comparison')
    parser.add_argument('--stages', default='', help='Optional stages, e.g. 0|1')
    args = parser.parse_args()
    stages = None
    if args.stages:
        stages = [[int(class_id) for class_id in stage.split(',') if class_id.strip()] for stage in args.stages.split('|')]
    run_comparison(args.dataset, args.rounds, args.epochs, args.seeds, args.output_dir, stages=stages)