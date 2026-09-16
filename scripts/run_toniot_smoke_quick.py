import sys
sys.path.insert(0, '.')
from experiments.runner import run_experiment

if __name__ == '__main__':
    run_experiment('TON_IoT datasets', rounds=1, epochs=1, exp_name='TON_IoT_smoke_quick', device='cpu', compress=False, agg_strategy='adaptive')
