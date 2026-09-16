**Adaptive Federated Aggregation — Implementation Plan**

**Overview**
- **Goal**: Implement an Adaptive Federated Aggregation mechanism within the Federated Class Incremental Learning (FCIL) framework from the base paper and evaluate on TON_IoT, IoT-NID, and Edge-IIoTset.
- **Success Criteria**: Improved global detection metrics (Accuracy, Precision, Recall, F1) under non-IID client distributions versus FedAvg/FedProx/FedDyn, with measured communication overhead.

**Phase 1 — Setup & Reproduction**
- **Repo scaffold**: Create folders `data/`, `clients/`, `server/`, `experiments/`, `results/`, `scripts/`.
- **Reproduce baseline**: Implement or import the base FCIL pipeline (local CIL on clients + federated aggregation) and verify baseline metrics on a small subset of TON_IoT.
- **Deliverable**: runnable baseline script `scripts/run_baseline.py` and baseline results in `results/baseline/`.
- **Acceptance**: Baseline reproduces reported trends (comparable accuracy/F1 within reasonable deviation).

**Phase 2 — Design Adaptive Aggregator**
- **Design doc**: Define adaptive aggregation strategy options: weighted aggregation by client utility, staleness-aware weighting, performance-based scaling, and client clustering.
- **Selection criterion**: Choose 1 primary strategy (e.g., hybrid weight = alpha * performance + beta * data_size + gamma * staleness) and fallback heuristics.
- **Deliverable**: `docs/aggregator_design.md` with math and pseudocode.

**Phase 3 — Implementation**
- **Server-side**: Implement `server/aggregator.py` exposing `aggregate(models, metadata)` with selectable strategies and logging.
- **Client-side**: Extend `clients/client.py` to compute and send metadata (local validation score, class-counts, compute budget, timestamp).
- **CIL integration**: Ensure CIL methods (MEMENTO+, iCaRL+) maintain prototypes or rehearsal buffers; adapt aggregation to model parameter formats.
- **Unit tests**: Add tests for aggregator weighting logic in `tests/test_aggregator.py`.

**Phase 4 — Datasets & Preprocessing**
- **Datasets**: Download TON_IoT, IoT-NID, Edge-IIoTset into `data/raw/` and convert to unified feature schema in `data/processed/`.
- **Preprocessing**: Normalization, categorical encoding, train/validation splits per-client, and incremental-class splits for CIL.
- **Scripts**: `scripts/prepare_data.py` and dataset readme `data/README.md`.

**Phase 5 — Experiments & Simulation**
- **Client heterogeneity**: Create simulation scenarios: IID, mild non-IID, severe non-IID, class-skewed, and compute/communication-limited clients.
- **Runs**: For each scenario, run experiments with aggregation: FedAvg, FedProx, FedDyn, and Adaptive aggregator. Use identical seeds and split-folds.
- **Automation**: `experiments/runner.py` to launch experiments, log with structured JSON into `results/`.

**Phase 6 — Evaluation & Analysis**
- **Metrics**: Compute Accuracy, Precision, Recall, F1, per-class performance, and communication overhead (bytes, rounds).
- **Ablations**: Vary alpha/beta/gamma terms, effect of metadata noise, and client dropouts.
- **Visualization**: Plot learning curves, per-class confusion matrices, and communication vs. performance trade-offs in `results/plots/`.
- **Deliverable**: `results/report.pdf` or notebook `results/analysis.ipynb`.

**Phase 7 — Optimization & Robustness**
- **Communication**: Implement optional compression (quantization/pruning) and client sampling schedules to reduce overhead.
- **Robustness**: Add defenses for malicious/Byzantine clients (clipping, median-based aggregation) as optional modes.
- **Acceptance**: Adaptive aggregator maintains performance under client dropout and noisy metadata.

**Phase 8 — Documentation & Delivery**
- **Code docs**: README with setup, dataset instructions, and `scripts/run_experiment.py` usage.
- **Paper/report**: Draft project report comparing with base paper, include methodology, experiments, and conclusions.
- **Presentation**: Update slides with results and timeline.

**Milestones & Timeline (suggested)**
- Week 1: Repo scaffold, reproduce baseline on small dataset.
- Week 2: Design aggregator and data preprocessing.
- Week 3: Implement aggregator and integrate CIL clients.
- Week 4: Run primary experiments, basic evaluation.
- Week 5: Ablations, optimizations, and documentation.

**Required Resources**
- Python 3.9+, PyTorch/TensorFlow, scikit-learn, pandas, matplotlib, seaborn.
- GPUs for faster training (optional but recommended).

**Quick Next Steps**
- **Action**: I'll create the repository scaffold and add baseline runner. Do you want me to start with reproducing the baseline now?

