# Adaptive Federated Class-Incremental IoT Intrusion Detection System
## Complete Project Architecture, Dataset Pipeline, Training & Web Interface Guide

---

## Table of Contents
1. [Project Overview & System Architecture](#1-project-overview--system-architecture)
2. [Environment Setup & Installation](#2-environment-setup--installation)
3. [Dataset Preparation & Directory Structure](#3-dataset-preparation--directory-structure)
   - [Where to Place Real Datasets (e.g. TON_IoT)](#where-to-place-real-datasets-eg-ton_iot)
   - [Preprocessing Commands & Partitioning](#preprocessing-commands--partitioning)
4. [Training Models (Federated & Class-Incremental Learning)](#4-training-models-federated--class-incremental-learning)
   - [Running CIL & Multi-Seed Experiments](#running-cil--multi-seed-experiments)
5. [Model Checkpoint & Artifact Storage](#5-model-checkpoint--artifact-storage)
   - [Where the Global Model is Saved](#where-the-global-model-is-saved)
   - [How Local Edge Models Work & Storage Details](#how-local-edge-models-work--storage-details)
   - [Experiment Logs & Generated Plots](#experiment-logs--generated-plots)
6. [Simulating & Testing with PCAP Files](#6-simulating--testing-with-pcap-files)
   - [Generating Sample PCAP Files](#generating-sample-pcap-files)
   - [CLI PCAP Feature Replay](#cli-pcap-feature-replay)
7. [Running the Live Web Visual Dashboard](#7-running-the-live-web-visual-dashboard)
   - [Launching the Web Server](#launching-the-web-server)
   - [Dashboard Controls & Features](#dashboard-controls--features)
8. [Conceptual Clarification: Global Model vs. Local Client Models in Deployment](#8-conceptual-clarification-global-model-vs-local-client-models-in-deployment)

---

## 1. Project Overview & System Architecture

This project implements a **Decentralized, Adaptive Federated Class-Incremental Learning Intrusion Detection System (IDS)** specifically designed for resource-constrained Internet of Things (IoT) network environments.

### Core Architectural Pillars

```
+-------------------------------------------------------------------+
|                        CENTRAL SERVER                             |
|  - Adaptive Aggregator (Validation Quality & staleness weighting) |
|  - Global Model Checkpoint Storage (results/<exp>/global_model.pt) |
+-------------------------------------------------------------------+
                                 ^
                                 | Model State Updates
                                 v
+-----------------------+ +-----------------------+ +-----------------------+
|  IoT Edge Gateway 1   | |  IoT Edge Gateway 2   | |  IoT Edge Gateway N   |
| - Local Telemetry     | | - Local Telemetry     | | - Local Telemetry     |
| - SimpleNet Training  | | - SimpleNet Training  | | - SimpleNet Training  |
| - Rehearsal Buffer    | | - Rehearsal Buffer    | | - Rehearsal Buffer    |
| - Teacher Distill Loss| | - Teacher Distill Loss| | - Teacher Distill Loss|
+-----------------------+ +-----------------------+ +-----------------------+
```

1. **Privacy-Preserving Federated Learning (FL)**: Edge gateways train local models on network flow telemetry. Only neural network weight updates (`state_dict`) are shared with the central server; raw network traffic never leaves the edge node.
2. **Adaptive Aggregation (`server/aggregator.py`)**: Unlike standard `FedAvg` which weights updates purely by data volume, our adaptive aggregator dynamically weights client contributions based on validation accuracy, dataset size, and model staleness.
3. **Class-Incremental Learning (CIL)**: When new attack classes emerge over time, local clients preserve performance on legacy attack types using a dual strategy:
   - **Rehearsal Buffer (`clients/client.py`)**: Stores a small exemplar set of historical flow samples (`rehearsal_capacity=200`).
   - **Knowledge Distillation Loss**: Uses the previous round's global model as a teacher model, applying KL-divergence loss to prevent catastrophic forgetting.
4. **On-the-Fly Feature Extraction & PCAP Replay (`scripts/pcap_replay.py`)**: Converts raw `.pcap` packet headers (IP, TCP/UDP ports, flags, TTL, length, inter-arrival time) into 20-dimensional feature vectors fed directly to the trained model.
5. **Real-Time Visual Web Dashboard (`web/server.py`)**: Serves an interactive SSE streaming web UI for live intrusion telemetry and threat alerts.

---

## 2. Environment Setup & Installation

### Prerequisites
- **Python 3.10+** (Tested on Python 3.10 / 3.11 / 3.14)
- **PyTorch 2.0+**
- **Scapy 2.5+** (for PCAP header parsing)

### Setup Instructions

```powershell
# 1. Clone or navigate to project directory
cd c:\Users\flow\Documents\coding\major_project

# 2. Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# 3. Install core dependencies
pip install -r requirements.txt
pip install scapy
```

---

## 3. Dataset Preparation & Directory Structure

### Where to Place Real Datasets (e.g. TON_IoT)

Place raw dataset files (CSV / PCAP / JSON) under the `data/raw/` folder:

```text
major_project/
├── data/
│   ├── raw/
│   │   └── TON_IoT/                <-- Place raw TON_IoT CSV files here
│   │       ├── Train_Test_Network.csv
│   │       └── Processed_IoT_dataset.csv
│   └── processed/                  <-- Processed output directory
│       ├── synthetic/              <-- Synthetic dataset
│       └── TON_IoT datasets/       <-- Preprocessed TON_IoT client partitions
│           ├── meta.npz             <-- Feature names & class labels metadata
│           └── clients/
│               ├── client_0.npz
│               ├── client_1.npz
│               └── client_N.npz
```

### Preprocessing Commands & Partitioning

#### Option A: Synthetic Dataset (Fastest for testing)
```powershell
python scripts/generate_synthetic_processed.py --n-clients 5 --dataset synthetic
```

#### Option B: Real Benchmark Datasets (TON_IoT)
To convert raw CSV dataset files into non-IID client partitions (`client_0.npz` ... `client_N.npz`):

```powershell
# Process full TON_IoT dataset into 5 client partitions
python scripts/prepare_data.py --raw-dir data/raw/TON_IoT --dataset "TON_IoT datasets" --n-clients 5

# Create a small TON_IoT sample dataset for quick smoke testing
python scripts/create_toniot_small.py
```

Each `client_N.npz` contains:
- `X`: $(N_{samples}, 20)$ normalized numerical flow features.
- `y`: $(N_{samples},)$ integer class labels (`0`: Normal, `1`: Attack Class 1, etc.).

---

## 4. Training Models (Federated & Class-Incremental Learning)

### Running Experiments

#### 1. Standard Baseline Experiment
Runs basic federated rounds across clients using specified aggregator (`adaptive`, `fedavg`, `weighted`, `fedprox`):

```powershell
python experiments/runner.py --dataset synthetic --rounds 5 --epochs 1 --exp synthetic_adaptive --aggregator adaptive
```

#### 2. Class-Incremental Learning Demonstration
Simulates stage-by-stage introduction of new threat classes (`--stages "0|1,2"`):

```powershell
python scripts/run_cil_demo.py --dataset synthetic --rounds 2 --exp cil_demo
```

#### 3. Multi-Seed Benchmark Comparison
Executes identical FL runs across 4 aggregation strategies (`adaptive`, `fedavg`, `weighted`, `fedprox`) over multiple random seeds (0, 1, 2) and outputs a aggregate summary:

```powershell
python scripts/run_multi_seed_comparison.py --dataset synthetic --rounds 3 --seeds 0 1 2 --output-dir results/multi_seed_comparison
```

---

## 5. Model Checkpoint & Artifact Storage

### Where the Global Model is Saved

When `experiments/runner.py` finishes the requested aggregation rounds, the aggregated global model checkpoint is automatically saved to disk:

```text
results/
├── cil_demo/
│   ├── global_model.pt     <-- TRAINED GLOBAL MODEL CHECKPOINT (PyTorch state_dict)
│   └── log.json            <-- Round-by-round evaluation log
├── multi_seed_comparison_adaptive_seed0/
│   ├── global_model.pt
│   └── log.json
└── plots/                  <-- Generated slide graphics (.png)
    ├── accuracy_over_rounds.png
    ├── forgetting_bar_chart.png
    └── aggregation_weights.png
```

### How Local Edge Models Work & Storage Details

1. **Local Model Instances**: During federated training, each `Client` instance in `clients/client.py` initializes a local `SimpleNet` instance in memory.
2. **Local Rehearsal Buffer**: Each client retains up to 200 historical `(x, y)` tensor samples in memory (`self.rehearsal`) to mix with new stage data during training.
3. **Aggregation Flow**:
   - Clients send their local `state_dict` parameters to `server/aggregator.py`.
   - The central aggregator computes weighted averages:
     $$W_{global} = \sum_{i=1}^{N} \alpha_i \cdot W_{local, i}$$
   - The final aggregated `W_{global}` is loaded into `global_model` and written to `results/<exp>/global_model.pt`.

---

## 6. Simulating & Testing with PCAP Files

### Generating Sample PCAP Files

To create a realistic `.pcap` capture file containing synthetic network traffic (HTTP/MQTT telemetry, TCP SYN floods, Port Scans, and SQLi injections):

```powershell
python scripts/generate_sample_pcap.py --output data/sample_traffic.pcap --count 30
```

### CLI PCAP Feature Replay

To stream `.pcap` packets through feature extraction and classify them live in the terminal:

```powershell
python scripts/pcap_replay.py --pcap data/sample_traffic.pcap --exp cil_demo --delay 0.5
```

---

## 7. Running the Live Web Visual Dashboard

### Launching the Web Server

Start the interactive dashboard server (listening on port `8501`):

```powershell
python web/server.py --port 8501
```

Open your browser to: **`http://localhost:8501/`**

### Dashboard Controls & Features

- **Live Telemetry Table**: Real-time table streaming packet summaries, timestamp, verdict (`[NORMAL TRAFFIC]` green vs `[IDS ALERT: <ATTACK>]` red), and model confidence.
- **Threat Level Badge**: Glowing header indicator switching dynamically between `NORMAL TRAFFIC` and `IDS ALERT`.
- **Softmax Probability Chart**: Live Chart.js bar graph displaying prediction probabilities across all classes for each packet.
- **Threat Distribution Donut Chart**: Real-time aggregate count of benign vs threat detections.
- **Interactive Controls**:
  - **Pause / Play Toggle**: Freeze or resume real-time packet streaming.
  - **Replay Delay Slider**: Adjust streaming speed from `0.1s` to `2.0s` per packet.
  - **Source Selector**: Switch seamlessly between PCAP packet captures (`data/sample_traffic.pcap`) and curated records (`data/demo_records.npz`).
  - **Audio Mute/Unmute**: Toggle sound alert beeps triggered on high-priority threat detections.

---

## 8. Conceptual Clarification: Global Model vs. Local Client Models in Deployment

### Should the Global Model or Local Client Models be Tested in the Web Interface?

> **Core Answer: The GLOBAL MODEL (`global_model.pt`) MUST be tested and deployed in the Web Interface.**

### Detailed Justification & Engineering Context

1. **Collective Intelligence**:
   - Individual edge client models only see their own local telemetry partition (which is often non-IID and restricted to specific devices or attack types).
   - The **Global Model** is the product of federated aggregation. It unifies threat intelligence across all edge gateways while benefiting from teacher distillation and rehearsal memory.
2. **Superior Generalization**:
   - Evaluating a single client model on multi-class network traffic would result in false positives or missed alerts for attack types that particular client never encountered locally.
   - The Global Model delivers high classification accuracy across all known and newly introduced threat classes.
3. **Real-World Deployment Architecture**:
   - In production IoT security architectures, the central server aggregates client updates into the Global Model.
   - Once aggregated, the updated Global Model checkpoint (`global_model.pt`) is compiled and deployed back to edge security gateways for real-time traffic inspection.
   - Therefore, the web dashboard simulates a central/gateway threat monitoring node running the **Global Model**.

---

## Summary Checklist for Running Full Demonstration

```powershell
# Step 1: Preprocess dataset or generate synthetic clients
python scripts/generate_synthetic_processed.py --dataset synthetic

# Step 2: Run Federated CIL training & save global model
python scripts/run_cil_demo.py --dataset synthetic --exp cil_demo

# Step 3: Curate fixed demo record set & generate sample PCAP
python scripts/curate_demo_records.py --dataset synthetic
python scripts/generate_sample_pcap.py --output data/sample_traffic.pcap

# Step 4: Generate slide visual charts
python scripts/plot_results.py

# Step 5: Launch Web Visual Dashboard
python web/server.py --port 8501

# Step 6: Open browser at http://localhost:8501/
```
