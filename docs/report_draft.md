# Draft Report — Adaptive Federated Aggregation for Incremental IoT IDS

## Objective
Implement an Adaptive Federated Aggregation method within a Federated Class Incremental Learning (FCIL) framework and evaluate its performance under heterogeneous client distributions.

## Implementation Summary
- Modular Python code with client/server/experiments scaffolding.
- Adaptive aggregator weighting combines `data_size`, `val_score`, and `staleness`.
- Clients support rehearsal buffers and optional knowledge distillation.
- Prototype aggregation (iCaRL-like) implemented for prototype-based classification.

## Experiments (preliminary)
- Synthetic dataset experiments run for multiple aggregators: adaptive, fedavg, weighted.
- Metrics: global accuracy, prototype accuracy, macro Precision/Recall/F1, communication bytes.

### Corrected comparison run

Using the same synthetic dataset, seed (`42`), three federated rounds, one local epoch, and five clients:

| Strategy | Final accuracy | Final macro-F1 | Communication bytes/round |
|---|---:|---:|---:|
| Adaptive | 0.3770 | 0.1959 | 113,980 |
| FedAvg | 0.3490 | 0.2486 | 113,980 |
| Weighted | 0.3490 | 0.2486 | 113,980 |
| FedProx (`mu=0.1`) | 0.3470 | 0.2494 | 113,980 |

Interpretation: Adaptive produced the highest accuracy in this single run and visibly used non-uniform weights, but its lower macro-F1 indicates weaker balanced performance across classes. This is preliminary evidence only; multiple seeds, longer training, class-balanced metrics, and explicit incremental stages are required before claiming an overall improvement.

Comparison plots are available in `results/comparison_seed42_plots/`.

### Multi-seed validation

A two-seed validation run with two rounds produced the following final-metric means and population standard deviations:

| Strategy | Accuracy (mean +/- std) | Macro-F1 (mean +/- std) |
|---|---:|---:|
| Adaptive | 0.3665 +/- 0.0085 | 0.2106 +/- 0.0288 |
| FedAvg | 0.3705 +/- 0.0045 | 0.2026 +/- 0.0207 |
| Weighted | 0.3705 +/- 0.0045 | 0.2026 +/- 0.0207 |
| FedProx (`mu=0.1`) | 0.3740 +/- 0.0010 | 0.2007 +/- 0.0189 |

The validation result does not establish a universal winner: Adaptive has the best mean macro-F1, while FedProx has the best mean accuracy. At least three to five seeds, longer training, and a fixed primary metric are needed for the final claim.

## Next steps
- Run corrected experiments on TON_IoT, IoT-NID, and Edge-IIoTset where schemas are validated.
- Tune aggregator hyperparameters (alpha, beta, gamma) via grid search.
- Add multiple seeds, class-incremental old/new metrics, and client dropout scenarios.
- Implement robust aggregation (clipping / median) and real serialized communication measurement.
- Prepare final report and slides.
