# Implementation Details — Adaptive Federated Aggregation FCIL Pipeline

This document summarizes the implemented system (short) and then provides a detailed description of each component, interfaces, and notes for extension.

## Short Summary
- Language: Python 3.x (tested on 3.14)
- Frameworks: PyTorch (models, training), numpy/pandas (data), scikit-learn (metrics, preprocessing), matplotlib/seaborn (plots)
- Architecture: Central server simulation + multiple local clients. Clients perform Class-Incremental Learning (CIL) support via rehearsal buffers and optional distillation. Server performs adaptive aggregation of client state_dicts.
- Key components implemented:
  - `clients/client.py` — local training, rehearsal buffer, prototype computation, FedProx support
  - `server/aggregator.py` — adaptive weighting (data_size, val_score, staleness) plus `fedavg` and `weighted` modes
  - `server/compression.py` — quantize/dequantize helpers and estimated bytes
  - `experiments/runner.py` — reproducible orchestration, class-stage filtering, evaluation, CLI flags, prototype aggregation, communication estimation
  - `scripts/prepare_data.py` — CSV ingestion, numeric feature selection, label factorization, IID/Dirichlet partitioning
  - `scripts/generate_synthetic_processed.py` — quick synthetic processed dataset generator
  - `scripts/plot_results.py` — load logs and produce standard plots
  - `scripts/run_multi_seed_comparison.py` — run identical strategy comparisons across multiple seeds and write mean/std summaries

## Detailed Implementation

### Clients (`clients/client.py`)
- Class `Client(client_id, model, device='cpu')` wraps a model copy and manages rehearsal and metadata.
- `local_train(train_loader, epochs, lr, teacher_state_dict=None, distill_lambda=0.5, mu=0.0, global_state=None)`:
  - Trains a local copy of the model using `SGD` and cross-entropy.
  - Supports optional knowledge distillation by loading `teacher_state_dict` into a teacher model and applying KLDiv loss.
  - Rehearsal buffer: `sample_rehearsal_loader()` and `_update_rehearsal_from_loader()` maintain a small replay buffer used during training to mitigate class-incremental forgetting.
  - FedProx support: when `mu>0` and `global_state` provided, adds proximal term `(mu/2) * ||w - w_global||^2` to the loss.
  - Returns `(state_dict, metadata)` where `metadata` contains `accuracy`, `data_size`, and `timestamp`.

- `get_features(x) / compute_prototypes(loader)` extract penultimate-layer features (works with `SimpleNet` which stores layers in `.net`) and compute per-class mean prototype vectors.

### Server Aggregator (`server/aggregator.py`)
- Provides `aggregate(client_states, metadatas, strategy='adaptive')`.
- `adaptive` strategy computes per-client weights combining:
  - `data_size` (client sample count),
  - `val_score` (if provided in metadata), and
  - `staleness` or recency (timestamp-based),
  with configurable hyperparameters (documented in `docs/aggregator_design.md`).
- Also supports `fedavg` (simple average) and `weighted` (by data_size) baselines.
- `compute_weights(...)` exposes the normalized weights for diagnostics; each experiment log now records `aggregation_weights` per round.

### Compression (`server/compression.py`)
- `quantize_state_dict(state_dict, dtype=torch.float16)` returns a copy with tensors converted to `float16` for simulated send size reduction.
- `dequantize_state_dict(state_dict, dtype=torch.float32)` converts them back for aggregation.
- `quantize_state_dict_bytes(state_dict, dtype)` estimates bytes after quantization (uses element count × dtype bits).

### Experiments Runner (`experiments/runner.py`)
- Orchestrates federated rounds and evaluation. Key responsibilities:
  - Load processed per-client `.npz` files from `data/processed/<dataset>/clients/`.
  - Instantiate `Client` objects with fresh model copies (`models.simple_nn.SimpleNet`).
  - For each round: broadcast global model (as teacher), run `local_train` on each client, collect state_dicts & metadata, optionally simulate quantization, call `aggregate(...)`, update global model, compute prototypes, evaluate global accuracy and prototype accuracy, log metrics and communication bytes to `results/<exp>/log.json`.
  - Uses deterministic Python, NumPy, and PyTorch seeds and creates a held-out local validation split so `val_score` is available to adaptive aggregation.
  - Optional class stages filter each client's current training/validation data while the client rehearsal buffer persists across rounds. Example: `--stages 0|1`.
- CLI options exposed: `--dataset`, `--rounds`, `--epochs`, `--exp`, `--device`, `--compress`, `--aggregator`, `--mu`, `--seed`, `--stages`.

### Data preparation (`scripts/prepare_data.py`)
- Reads CSVs under `data/raw/<dataset>/` recursively.
- Auto-detects label column from common names or falls back to the last column.
- Drops rows with missing labels, factorizes string labels to integer ids, selects numeric features (or one-hot-encodes categoricals when no numeric columns present), standard-scales features, partitions to clients using IID or Dirichlet splits, and writes per-client compressed `.npz` files to `data/processed/<dataset>/clients/` and metadata to `data/processed/<dataset>/meta.npz`.

### Scripts and Utilities
- `scripts/generate_synthetic_processed.py` — produce toy per-client `.npz` files for quick tests.
- `scripts/plot_results.py` — loads experiment logs and writes PNG plots for `global_accuracy`, `prototype_accuracy`, `communication_bytes`, and `f1_macro`.
- `scripts/run_toniot_smoke.py` / `run_toniot_small_*` — helper runners used during development to avoid quoting/CLI complexities.

### Models
- `models/simple_nn.py` — a small MLP used as default model for simulations. Easy to swap for larger architectures.

### Tests
- Unit tests exist for aggregation behavior and tensor handling under `tests/` (e.g., `test_aggregator.py`). Run with `pytest`.

### Results layout
- `results/<exp>/log.json` — per-round metrics with entries containing `round`, `global_accuracy`, `precision_macro`, `recall_macro`, `f1_macro`, `prototype_accuracy`, `communication_bytes`, `aggregation_weights`, `class_stage`, `class_accuracy`, `stage_metrics`, and `metadata_summary`.
- `results/<exp>/plots/` — PNG output from `scripts/plot_results.py`.

## Notes, Limitations, and Next Steps
- Communication measurement is currently simulated by quantized-size estimation. Implement real serialization + compression (e.g., `pickle`/`msgpack` + `lz4`) to measure actual bytes and wall-clock transfer time.
- CIL support currently includes explicit class-stage filtering, rehearsal, distillation, and prototypes (iCaRL-inspired), but not a complete paper-faithful iCaRL+ or MEMENTO+ implementation.
- Aggregator hyperparameters (`alpha`, `beta`, `gamma` governing data_size/val_score/staleness) should be tuned experimentally; use the runner to sweep values and save results in `results/`.
- Add deterministic `configs/*.yaml` and experiment manifests for reproducibility.

## Verified Status (28 August 2026)

### Completed and demonstrated
- Reproducible seeded runner with held-out local validation splits.
- Adaptive aggregation using non-uniform weights derived from client data size, validation score, and update recency.
- FedAvg, data-weighted aggregation, and FedProx comparison modes.
- Optional class-stage filtering through `--stages`, with rehearsal buffers retained between stages.
- Per-round aggregation-weight logging, accuracy, macro precision/recall/F1, prototype accuracy, and communication-size estimates.
- Repeatable demo command: `python scripts/run_cil_demo.py`.
- Corrected three-round synthetic comparison and plots in `results/comparison_seed42_plots/`.

### Corrected comparison evidence

The same-seed synthetic comparison used five clients, three rounds, and one local epoch:

| Strategy | Final accuracy | Final macro-F1 | Communication bytes/round |
|---|---:|---:|---:|
| Adaptive | 0.3770 | 0.1959 | 113,980 |
| FedAvg | 0.3490 | 0.2486 | 113,980 |
| Weighted | 0.3490 | 0.2486 | 113,980 |
| FedProx (`mu=0.1`) | 0.3470 | 0.2494 | 113,980 |

Adaptive achieved the highest accuracy in this run, but the lowest macro-F1. Therefore, the current evidence demonstrates that the adaptive mechanism changes aggregation and can improve majority-weighted accuracy, but does not establish overall superiority. Longer runs, multiple seeds, class-balanced objectives, and per-class/old-new metrics are required.

### Multi-seed validation

The validation matrix used two seeds, two rounds, one local epoch, and the same synthetic dataset for all four strategies:

| Strategy | Mean accuracy | Accuracy std | Mean macro-F1 | Macro-F1 std |
|---|---:|---:|---:|---:|
| Adaptive | 0.3665 | 0.0085 | 0.2106 | 0.0288 |
| FedAvg | 0.3705 | 0.0045 | 0.2026 | 0.0207 |
| Weighted | 0.3705 | 0.0045 | 0.2026 | 0.0207 |
| FedProx (`mu=0.1`) | 0.3740 | 0.0010 | 0.2007 | 0.0189 |

This short validation shows Adaptive had the highest mean macro-F1, while FedProx had the highest mean accuracy. It is not statistically conclusive because only two seeds and two rounds were used. The reusable command is `python scripts/run_multi_seed_comparison.py --dataset synthetic --rounds 3 --seeds 0 1 2`.

### Explicit non-claims
- This is a centralized simulation, not a live multi-device federated deployment.
- Float16 communication values are estimates, not measurements from an actual network transfer.
- The project does not yet implement full MEMENTO+, iCaRL+, FedDyn, differential privacy, secure aggregation, live packet capture, or zero-day detection.

## Developer notes
- When adding a new dataset, ensure `scripts/prepare_data.py` correctly identifies the label column and numerical features; if the dataset needs special parsing, implement a dataset-specific reader and call the existing partition utilities.
- Follow the existing code style; new functionality should include unit tests under `tests/`.

---
Generated on: 2026-08-28
