import sys
sys.path.insert(0, '.')
from scripts.plot_results import plot_experiments

if __name__ == '__main__':
    plot_experiments([
        'results/TON_IoT_small_adaptive',
        'results/TON_IoT_small_fedavg',
        'results/TON_IoT_small_weighted',
        'results/TON_IoT_small_fedprox',
    ], out_dir='results/TON_IoT_small_plots')
