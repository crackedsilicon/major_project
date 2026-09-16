Adaptive Federated Aggregation for Incremental IoT IDS

Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Prepare datasets:

```bash
python scripts/prepare_data.py --dataset TON_IoT --out data/processed
```

3. Run baseline (stub):

```bash
python scripts/run_baseline.py
```

See `IMPLEMENTATION_PLAN.md` for full project plan.

Run experiments using processed datasets (see `scripts/prepare_data.py` and `scripts/generate_synthetic_processed.py`):

```bash
python experiments/runner.py --dataset synthetic --rounds 5 --epochs 1 --exp synthetic_exp
```

Run the short class-incremental demonstration:

```bash
python scripts/run_cil_demo.py
```

The demo exposes class `0` in the first round and class `1` in the second round, preserves client rehearsal buffers, and logs adaptive aggregation weights plus `old_accuracy`, `new_accuracy`, `seen_accuracy`, and `forgetting` in `results/cil_demo/log.json`.

Run a same-settings multi-seed comparison:

```bash
python scripts/run_multi_seed_comparison.py --dataset synthetic --rounds 3 --seeds 0 1 2
```

The summary is written to `results/multi_seed_comparison/summary.json`.