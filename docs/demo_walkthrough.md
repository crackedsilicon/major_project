# Adaptive Federated IoT IDS: 5-7 Minute Presentation Walkthrough & Fallback Guide

This guide provides a structured 5-7 minute presentation script, technical commentary, PCAP replay instructions, and emergency fallback protocols for demonstrating the Adaptive Federated Class-Incremental IoT Intrusion Detection System.

---

## 1. Executive Summary & Plain-English Framing (1 Min)

> **Framing Analogy: Airport Security Inspection**
> *"Imagine an international airport network: rather than sending every passenger's full baggage data to a central headquarters (which violates privacy and creates bandwidth bottlenecks), local security checkpoints independently screen traffic and share only security pattern updates. Our project brings this privacy-preserving model to IoT networks: edge gateways locally inspect network traffic, collaboratively aggregate model updates via Adaptive Federated Learning, and dynamically adapt when brand new attack types emerge."*

---

## 2. 6-Part Narrative Talk Track (5-7 Mins Total)

### Part 1: Architecture & Federated Setup
- **Talk Track**: *"We operate a decentralized network of IoT edge clients running PyTorch lightweight neural network models (`SimpleNet`). Raw network telemetry never leaves the local edge node. Aggregation occurs at the central server using an adaptive weighting strategy based on validation quality, sample volume, and model staleness."*
- **Key Visual**: Architecture diagram showing Edge Clients $\leftrightarrow$ Central Aggregator.

---

### Part 2: Normal Telemetry Baseline Ingestion
- **Talk Track**: *"First, let's observe normal IoT device communications. As traffic flows through the edge checkpoint, the global model predicts benign operation."*
- **Live Demo Option A (CLI)**: Run `python scripts/demo_inference.py --dataset synthetic --records-path data/demo_records.npz --delay 0.5`
- **Live Demo Option B (Web Dashboard)**: Open `http://localhost:8501/` in any browser.

---

### Part 3: Live PCAP Traffic Ingestion & IDS Alert Output
- **Talk Track**: *"Now, let's replay a real network capture (`data/sample_traffic.pcap`) containing TCP SYN floods, port scans, and malicious payload injections. The model extracts flow features on the fly and triggers real-time alerts."*
- **CLI PCAP Execution**: Run `python scripts/pcap_replay.py --pcap data/sample_traffic.pcap --exp cil_demo --delay 0.5`
- **Expected Terminal Output**:
  ```text
  [04/30] Ether / IP / TCP 10.208.223.2:50753 >... -> [IDS ALERT: THREAT CLASS 1] (Conf: 35.7%)
  [15/30] Ether / IP / TCP 45.33.32.156:54321 >... -> [IDS ALERT: THREAT CLASS 1] (Conf: 35.7%)
  ```

---

### Part 4: Dynamic Adaptive Weighting
- **Talk Track**: *"Why do we trust one edge client over another? Standard FedAvg treats noisy, small, or outdated clients equally. Our Adaptive Aggregator dynamically scores clients each round based on validation accuracy and data volume, giving higher weight to reliable edge nodes."*
- **Key Visual**: `results/plots/aggregation_weights.png` showing client weight trajectories.

---

### Part 5: Class-Incremental Learning & Forgetting Mitigation
- **Talk Track**: *"In real IoT environments, new attack vectors appear over time. When new attack classes are introduced in Stage 2, traditional models suffer from catastrophic forgetting. Our system uses a lightweight rehearsal buffer and distillation loss to retain high accuracy on legacy attacks while mastering newly observed threats."*
- **Key Visual**: `results/plots/forgetting_bar_chart.png` demonstrating retained accuracy vs minimal forgetting.

---

### Part 6: Method Comparison & Performance Summary
- **Talk Track**: *"Across multi-seed experiments on benchmark IoT datasets, our Adaptive scheme consistently outperforms standard FedAvg, Weighted FL, and FedProx in classification macro F1 score and communication efficiency."*
- **Key Visual**: `results/plots/accuracy_over_rounds.png` and `results/plots/f1_macro.png`.

---

## 3. Live Demonstration Execution Commands

### A. Terminal CLI IDS Alert Demo
```powershell
# 1. Run live offline alert demo
python scripts/demo_inference.py --dataset synthetic --exp cil_demo --records-path data/demo_records.npz --delay 0.8

# 2. Run PCAP packet replay engine
python scripts/pcap_replay.py --pcap data/sample_traffic.pcap --exp cil_demo --delay 0.5
```

### B. Interactive Web Visual Dashboard
```powershell
# Launch the web dashboard server
python web/server.py --port 8501

# Open http://localhost:8501/ in browser
```

---

## 4. Fallback & Backup Protocol

If live execution encounters environment issues during presentation:

1. **Pre-Generated Visuals**: Refer directly to static PNG charts in `results/plots/`:
   - `accuracy_over_rounds.png`
   - `forgetting_bar_chart.png`
   - `aggregation_weights.png`
2. **Curated Demo Records & PCAP Manifest**: Show `data/demo_records.npz` and `data/sample_traffic.pcap` as empirical evidence.
3. **Pre-Executed Log Output**: Show saved terminal output logs or static HTML view of `http://localhost:8501/`.
