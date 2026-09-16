#!/usr/bin/env python
"""
web/server.py — Flask-based backend for the Adaptive Federated IoT IDS Dashboard.

Provides:
  - Static file serving for the SPA frontend
  - /api/upload — CSV/PCAP file upload + immediate classification
  - /api/stream — SSE live streaming of demo records or PCAP replay
  - /api/training-results — Experiment log data for visualization
  - /api/model-info — Model architecture and metadata
  - /api/control — Playback control (pause, speed, source)

USAGE:
    python web/server.py --port 8501
"""

import argparse
import json
import io
import os
import sys
import time
import tempfile
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from models.simple_nn import SimpleNet
from sklearn.preprocessing import StandardScaler

# Try importing scapy for PCAP support
try:
    from scapy.all import rdpcap, IP, TCP, UDP, Raw
    from scripts.pcap_replay import extract_features_from_packet
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Flask App Setup
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path='')
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max upload

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
STATE = {
    "is_playing": True,
    "delay": 0.5,
    "dataset": "synthetic",
    "exp": "cil_demo",
    "pcap_file": "data/sample_traffic.pcap",
}

# Cache for loaded model
_model_cache = {}


def get_model(exp_name="cil_demo", input_dim=20, num_classes=3):
    """Load and cache the trained global model."""
    cache_key = f"{exp_name}_{input_dim}_{num_classes}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    checkpoint_path = PROJECT_ROOT / "results" / exp_name / "global_model.pt"
    model = SimpleNet(input_dim=input_dim, num_classes=num_classes)
    if checkpoint_path.exists():
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=False)
        except Exception:
            try:
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"Warning: Could not load checkpoint {checkpoint_path}: {e}")
    model.eval()
    _model_cache[cache_key] = model
    return model


def get_class_names(exp_name="cil_demo", dataset="synthetic"):
    """Load class names from metadata or return defaults."""
    defaults = ["Normal", "DDoS Attack", "Reconnaissance / Scan", "Injection Attack"]

    # Try demo records first
    demo_npz = PROJECT_ROOT / "data" / "demo_records.npz"
    if demo_npz.exists():
        try:
            d = np.load(demo_npz, allow_pickle=True)
            if "class_names" in d:
                return [str(n) for n in d["class_names"]]
        except Exception:
            pass

    # Try dataset metadata
    meta_path = PROJECT_ROOT / "data" / "processed" / dataset / "meta.npz"
    if meta_path.exists():
        try:
            m = np.load(meta_path, allow_pickle=True)
            for key in ("class_names", "label_values", "labels"):
                if key in m:
                    return [str(n) for n in m[key]]
        except Exception:
            pass

    return defaults


# ---------------------------------------------------------------------------
# Static File Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return send_from_directory(str(WEB_DIR), 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(str(WEB_DIR), filename)


# ---------------------------------------------------------------------------
# API: Model Info
# ---------------------------------------------------------------------------
@app.route('/api/model-info')
def model_info():
    """Return model architecture details and available experiments."""
    experiments = []
    results_dir = PROJECT_ROOT / "results"
    if results_dir.exists():
        for d in sorted(results_dir.iterdir()):
            if d.is_dir() and (d / "log.json").exists():
                experiments.append(d.name)

    model = get_model()
    param_count = sum(p.numel() for p in model.parameters())

    return jsonify({
        "model_name": "SimpleNet",
        "architecture": "MLP (Linear→ReLU→Linear→ReLU→Linear)",
        "param_count": param_count,
        "framework": "PyTorch",
        "experiments": experiments[:20],
        "current_exp": STATE["exp"],
        "scapy_available": SCAPY_AVAILABLE,
        "aggregation_strategies": ["adaptive", "fedavg", "weighted", "fedprox"],
        "cil_methods": ["Rehearsal Buffer", "Knowledge Distillation", "Prototype Aggregation"],
    })


# ---------------------------------------------------------------------------
# API: Training Results
# ---------------------------------------------------------------------------
@app.route('/api/training-results')
def training_results():
    """Return training log data for all available experiments."""
    exp_name = request.args.get('exp', None)
    results_dir = PROJECT_ROOT / "results"
    data = {}

    if exp_name:
        # Single experiment
        log_path = results_dir / exp_name / "log.json"
        if log_path.exists():
            with open(log_path) as f:
                data[exp_name] = json.load(f)
    else:
        # Aggregate key experiments for comparison
        priority_exps = [
            "cil_demo",
            "TON_IoT_small_adaptive", "TON_IoT_small_fedavg",
            "TON_IoT_small_fedprox", "TON_IoT_small_weighted",
            "comparison_seed42_adaptive", "comparison_seed42_fedavg",
            "comparison_seed42_fedprox", "comparison_seed42_weighted",
        ]
        for exp in priority_exps:
            log_path = results_dir / exp / "log.json"
            if log_path.exists():
                try:
                    with open(log_path) as f:
                        data[exp] = json.load(f)
                except Exception:
                    pass

        # If no priority experiments found, load whatever is available
        if not data and results_dir.exists():
            for d in sorted(results_dir.iterdir())[:10]:
                log_path = d / "log.json"
                if log_path.exists():
                    try:
                        with open(log_path) as f:
                            data[d.name] = json.load(f)
                    except Exception:
                        pass

    return jsonify(data)


# ---------------------------------------------------------------------------
# API: File Upload & Analysis
# ---------------------------------------------------------------------------
@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle CSV or PCAP file upload and run classification."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = file.filename.lower()
    model = get_model()
    class_names = get_class_names()
    num_classes = len(class_names)

    results = []

    try:
        if filename.endswith('.csv'):
            results = _process_csv_upload(file, model, class_names, num_classes)
        elif filename.endswith('.pcap') or filename.endswith('.pcapng'):
            if not SCAPY_AVAILABLE:
                return jsonify({"error": "Scapy not installed. Run: pip install scapy"}), 500
            results = _process_pcap_upload(file, model, class_names, num_classes)
        else:
            return jsonify({"error": f"Unsupported file type: {filename}. Use .csv or .pcap"}), 400

    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

    # Compute summary stats
    total = len(results)
    alerts = sum(1 for r in results if not r.get("is_normal", True))
    avg_conf = np.mean([r["confidence"] for r in results]) if results else 0

    # Class distribution
    class_dist = {}
    for r in results:
        label = r.get("pred_label", "Unknown")
        class_dist[label] = class_dist.get(label, 0) + 1

    return jsonify({
        "filename": file.filename,
        "total_records": total,
        "total_alerts": alerts,
        "alert_rate": (alerts / total * 100) if total > 0 else 0,
        "avg_confidence": float(avg_conf * 100),
        "class_distribution": class_dist,
        "results": results,
    })


def _process_csv_upload(file, model, class_names, num_classes):
    """Parse uploaded CSV, extract features, run classification."""
    content = file.read().decode('utf-8', errors='replace')
    df = pd.read_csv(io.StringIO(content), low_memory=False)

    # Try to find label column
    label_col = None
    for name in ['label', 'Label', 'attack', 'class', 'target', 'type', 'Attack_type']:
        if name in df.columns:
            label_col = name
            break
    if label_col is None:
        label_col = df.columns[-1]

    # Drop obvious non-feature columns
    drop_cols = [c for c in df.columns if any(kw in c.lower() for kw in ['time', 'timestamp', 'date'])]
    if 'id' in [c.lower() for c in df.columns]:
        drop_cols.extend([c for c in df.columns if c.lower() == 'id'])

    # Extract features
    feature_cols = df.drop(columns=[label_col] + [c for c in drop_cols if c in df.columns], errors='ignore')
    feature_cols = feature_cols.select_dtypes(include=[np.number])

    if feature_cols.shape[1] == 0:
        feature_cols = pd.get_dummies(df.drop(columns=[label_col], errors='ignore'))

    X = feature_cols.values.astype(np.float32)

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

    # Normalize
    scaler = StandardScaler()
    if X.shape[0] > 1:
        X = scaler.fit_transform(X)

    # Adjust dimensions to match model input
    model_input_dim = model.net[0].in_features
    if X.shape[1] < model_input_dim:
        pad = np.zeros((X.shape[0], model_input_dim - X.shape[1]), dtype=np.float32)
        X = np.hstack([X, pad])
    elif X.shape[1] > model_input_dim:
        X = X[:, :model_input_dim]

    # Get actual labels if available
    actual_labels = df[label_col].values if label_col in df.columns else None

    # Run classification
    results = []
    max_records = min(len(X), 500)  # Cap at 500 for performance
    model.eval()
    with torch.no_grad():
        for i in range(max_records):
            x_tensor = torch.tensor(X[i:i+1], dtype=torch.float32)
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).tolist()
            pred = int(logits.argmax(dim=1).item())

            pred_label = class_names[pred] if pred < len(class_names) else f"Class {pred}"
            actual = str(actual_labels[i]) if actual_labels is not None else None

            record = {
                "seq": i + 1,
                "summary": f"CSV Row #{i+1}" + (f" (Label: {actual})" if actual else ""),
                "source": "csv",
                "pred_class": pred,
                "pred_label": pred_label,
                "is_normal": pred == 0,
                "confidence": float(probs[pred]),
                "probabilities": [float(p) for p in probs],
                "features": [float(f) for f in X[i][:8]],
                "actual_label": actual,
                "timestamp": time.strftime("%H:%M:%S"),
            }
            results.append(record)

    return results


def _process_pcap_upload(file, model, class_names, num_classes):
    """Parse uploaded PCAP, extract features per packet, classify."""
    # Save to temp file for scapy
    with tempfile.NamedTemporaryFile(suffix='.pcap', delete=False) as tmp:
        file.save(tmp)
        tmp_path = tmp.name

    try:
        packets = rdpcap(tmp_path)
    finally:
        os.unlink(tmp_path)

    results = []
    prev_time = None
    max_packets = min(len(packets), 500)  # Cap at 500

    model.eval()
    with torch.no_grad():
        for i in range(max_packets):
            pkt = packets[i]
            feats, prev_time = extract_features_from_packet(pkt, prev_time)
            x_tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).tolist()
            pred = int(logits.argmax(dim=1).item())

            pred_label = class_names[pred] if pred < len(class_names) else f"Class {pred}"
            summary = pkt.summary() if hasattr(pkt, 'summary') else f"Packet #{i+1}"

            record = {
                "seq": i + 1,
                "total": len(packets),
                "summary": summary,
                "source": "pcap",
                "pred_class": pred,
                "pred_label": pred_label,
                "is_normal": pred == 0,
                "confidence": float(probs[pred]),
                "probabilities": [float(p) for p in probs],
                "features": [float(f) for f in feats[:8]],
                "timestamp": time.strftime("%H:%M:%S"),
            }
            results.append(record)

    return results


# ---------------------------------------------------------------------------
# API: SSE Live Stream
# ---------------------------------------------------------------------------
@app.route('/api/stream')
def sse_stream():
    """Server-Sent Events stream for live demo playback."""
    def generate():
        model = get_model()
        class_names = get_class_names()

        # Load PCAP packets
        packets = []
        pcap_path = PROJECT_ROOT / STATE['pcap_file']
        if SCAPY_AVAILABLE and pcap_path.exists():
            try:
                packets = rdpcap(str(pcap_path))
            except Exception:
                packets = []

        # Load curated demo records as fallback
        demo_npz = PROJECT_ROOT / "data" / "demo_records.npz"
        curated_X, curated_y = None, None
        if demo_npz.exists():
            try:
                d = np.load(demo_npz, allow_pickle=True)
                curated_X, curated_y = d["X"], d["y"]
                if "class_names" in d:
                    class_names = [str(n) for n in d["class_names"]]
            except Exception:
                pass

        # Also try ton_iot_demo_records
        toniot_demo = PROJECT_ROOT / "data" / "ton_iot_demo_records.npz"
        if curated_X is None and toniot_demo.exists():
            try:
                d = np.load(toniot_demo, allow_pickle=True)
                curated_X, curated_y = d["X"], d["y"]
                if "class_names" in d:
                    class_names = [str(n) for n in d["class_names"]]
            except Exception:
                pass

        packet_idx = 0
        curated_idx = 0
        prev_time = None

        while True:
            try:
                if not STATE['is_playing']:
                    time.sleep(0.2)
                    continue

                event_data = None

                if packets and packet_idx < len(packets):
                    pkt = packets[packet_idx]
                    feats, prev_time = extract_features_from_packet(pkt, prev_time)
                    x_tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)

                    with torch.no_grad():
                        logits = model(x_tensor)
                        probs = torch.softmax(logits, dim=1).squeeze(0).tolist()
                        pred = int(logits.argmax(dim=1).item())

                    summary = pkt.summary() if hasattr(pkt, 'summary') else f"Packet #{packet_idx+1}"
                    event_data = {
                        "seq": packet_idx + 1,
                        "total": len(packets),
                        "summary": summary,
                        "source": "pcap",
                        "pred_class": pred,
                        "pred_label": class_names[pred] if pred < len(class_names) else f"Class {pred}",
                        "is_normal": pred == 0,
                        "confidence": float(probs[pred]),
                        "probabilities": [float(p) for p in probs],
                        "features": [float(f) for f in feats[:8]],
                        "timestamp": time.strftime("%H:%M:%S"),
                    }
                    packet_idx = (packet_idx + 1) % len(packets)

                elif curated_X is not None and len(curated_X) > 0:
                    i = curated_idx % len(curated_X)
                    feats = curated_X[i]
                    actual = int(curated_y[i])
                    x_tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)

                    with torch.no_grad():
                        logits = model(x_tensor)
                        probs = torch.softmax(logits, dim=1).squeeze(0).tolist()
                        pred = int(logits.argmax(dim=1).item())

                    actual_label = class_names[actual] if actual < len(class_names) else f"Class {actual}"
                    event_data = {
                        "seq": i + 1,
                        "total": len(curated_X),
                        "summary": f"Flow #{i+1} (Target: {actual_label})",
                        "source": "curated",
                        "pred_class": pred,
                        "pred_label": class_names[pred] if pred < len(class_names) else f"Class {pred}",
                        "actual_class": actual,
                        "actual_label": actual_label,
                        "is_correct": pred == actual,
                        "is_normal": pred == 0,
                        "confidence": float(probs[pred]),
                        "probabilities": [float(p) for p in probs],
                        "features": [float(f) for f in feats[:8]],
                        "timestamp": time.strftime("%H:%M:%S"),
                    }
                    curated_idx += 1

                if event_data:
                    yield f"data: {json.dumps(event_data)}\n\n"

                time.sleep(max(0.1, float(STATE['delay'])))

            except GeneratorExit:
                break
            except Exception as e:
                print(f"SSE error: {e}")
                time.sleep(1.0)

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive'})


# ---------------------------------------------------------------------------
# API: Playback Control
# ---------------------------------------------------------------------------
@app.route('/api/control', methods=['POST'])
def control():
    """Update playback state."""
    data = request.get_json(silent=True) or {}
    for k, v in data.items():
        if k in STATE:
            STATE[k] = v
    return jsonify({"status": "ok", "state": STATE})


# ---------------------------------------------------------------------------
# API: Status
# ---------------------------------------------------------------------------
@app.route('/api/status')
def status():
    return jsonify(STATE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Adaptive Federated IoT IDS — Web Dashboard Server")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print("=" * 65)
    print("   Adaptive Federated IoT IDS - Web Visual Dashboard")
    print("=" * 65)
    print(f"   Server running at:  http://localhost:{args.port}/")
    print(f"   PCAP support:       {'[OK] Scapy loaded' if SCAPY_AVAILABLE else '[X] Install scapy'}")
    print(f"   Model checkpoint:   results/{STATE['exp']}/global_model.pt")
    print("=" * 65)

    app.run(host='0.0.0.0', port=args.port, debug=args.debug, threaded=True)


if __name__ == '__main__':
    main()
