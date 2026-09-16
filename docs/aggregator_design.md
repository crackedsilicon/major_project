# Adaptive Aggregator Design

## Overview

The aggregator computes a weighted average of client state_dicts where per-client
weights are computed from metadata. Metadata fields:

- `data_size` — number of local training samples
- `val_score` — local validation accuracy (or other utility metric)
- `timestamp` — last local training timestamp (used to penalize staleness)

Weights are computed as:

$$
w_i = \frac{(data\_size_i)^{\alpha} (1 + val\_score_i)^{\beta}}{1 + \gamma \cdot staleness_i}
$$

Normalized weights are then used to compute a parameter-wise weighted average of
PyTorch tensors from client `state_dict`s.

## Hyperparameters

- `alpha` (default 1.0): controls influence of local data size
- `beta` (default 1.0): controls influence of validation quality
- `gamma` (default 1.0): controls staleness penalty magnitude

These hyperparameters will be tuned via ablation experiments.

## Pseudocode

1. Collect `state_dict` and `metadata` from each selected client.
2. Compute per-client weights using the formula above.
3. For each parameter key, initialize accumulator tensor to zeros.
4. For each client, add `weight * param_tensor` to accumulator.
5. Return aggregated `state_dict` composed from accumulators.

## Extensions

- Cluster clients by data distribution and perform cluster-wise aggregation.
- Use performance-normalized weighting (e.g., dividing by local loss variance).
- Add robustness: clip per-parameter updates, or use median-based aggregation.

## Metrics for evaluation

- Global Accuracy, Precision, Recall, F1
- Per-class accuracy to detect catastrophic forgetting
- Communication overhead (bytes transmitted per round)
