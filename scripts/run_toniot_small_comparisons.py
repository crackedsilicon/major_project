import sys
sys.path.insert(0, '.')
from experiments.runner import run_experiment


def main() -> None:
    runs = [
        ('adaptive', False, 0.0),
        ('fedavg', False, 0.0),
        ('weighted', True, 0.0),
        ('fedprox', False, 0.1),
    ]
    for strategy, compress, mu in runs:
        exp_name = f'TON_IoT_small_{strategy}'
        run_experiment(
            'TON_IoT_small',
            rounds=2,
            epochs=1,
            exp_name=exp_name,
            device='cpu',
            compress=compress,
            agg_strategy=strategy,
            mu=mu,
        )


if __name__ == '__main__':
    main()
