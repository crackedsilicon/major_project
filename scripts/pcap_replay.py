#!/usr/bin/env python
"""
pcap_replay.py

Reads a saved .pcap network capture file using Scapy, extracts 20 numerical
flow telemetry features per packet on the fly, feeds them into the trained PyTorch
global model checkpoint, and outputs real-time classification alerts.

USAGE
-----
    python scripts/pcap_replay.py --pcap data/sample_traffic.pcap --exp cil_demo --delay 0.5
"""

import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    from models.simple_nn import SimpleNet
except ImportError:
    print("PyTorch is required. Ensure virtual environment is active.")
    sys.exit(1)

try:
    from scapy.all import rdpcap, IP, TCP, UDP, ICMP, Raw
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def extract_features_from_packet(pkt, prev_time=None):
    """
    Extracts a 20-element normalized feature vector from a Scapy packet.
    Features include packet length, header sizes, port numbers, flags,
    inter-arrival time, payload presence, and protocol indicators.
    """
    feats = np.zeros(20, dtype=np.float32)

    length = len(pkt)
    feats[0] = min(1.0, length / 1500.0)  # normalized pkt length

    pkt_time = float(getattr(pkt, 'time', time.time()))
    delta = (pkt_time - prev_time) if prev_time is not None else 0.05
    feats[1] = min(1.0, delta / 2.0)  # normalized inter-arrival delta

    if IP in pkt:
        ip = pkt[IP]
        feats[2] = min(1.0, ip.len / 1500.0)
        feats[3] = min(1.0, ip.ttl / 255.0)
        feats[4] = 1.0 if ip.proto == 6 else (0.5 if ip.proto == 17 else 0.0)

        # Hash IP address segments into features 5-8
        src_parts = [int(p) for p in ip.src.split('.')] if '.' in ip.src else [0]*4
        dst_parts = [int(p) for p in ip.dst.split('.')] if '.' in ip.dst else [0]*4
        feats[5] = src_parts[-1] / 255.0
        feats[6] = dst_parts[-1] / 255.0

    if TCP in pkt:
        tcp = pkt[TCP]
        feats[7] = min(1.0, tcp.sport / 65535.0)
        feats[8] = min(1.0, tcp.dport / 65535.0)
        # TCP Flags
        feats[9] = 1.0 if 'S' in tcp.flags else 0.0  # SYN
        feats[10] = 1.0 if 'A' in tcp.flags else 0.0 # ACK
        feats[11] = 1.0 if 'P' in tcp.flags else 0.0 # PSH
        feats[12] = 1.0 if 'F' in tcp.flags else 0.0 # FIN
        feats[13] = min(1.0, tcp.window / 65535.0)

    elif UDP in pkt:
        udp = pkt[UDP]
        feats[7] = min(1.0, udp.sport / 65535.0)
        feats[8] = min(1.0, udp.dport / 65535.0)
        feats[14] = min(1.0, udp.len / 1500.0)

    if Raw in pkt:
        payload = pkt[Raw].load
        feats[15] = min(1.0, len(payload) / 1000.0)
        # Simple payload byte characteristics
        feats[16] = 1.0 if b"GET" in payload or b"POST" in payload else 0.0
        feats[17] = 1.0 if b"MQTT" in payload or b"SELECT" in payload or b"OR" in payload else 0.0

    feats[18] = np.mean(feats[:10])
    feats[19] = np.std(feats[:10])

    return feats, pkt_time


def main():
    parser = argparse.ArgumentParser(description="Live PCAP Packet Replay and IDS Classification")
    parser.add_argument("--pcap", default="data/sample_traffic.pcap", help="Path to input .pcap file")
    parser.add_argument("--exp", default="cil_demo", help="Experiment name to load checkpoint from")
    parser.add_argument("--checkpoint", default=None, help="Path to global_model.pt")
    parser.add_argument("--delay", type=float, default=0.5, help="Replay pacing delay per packet in seconds")
    args = parser.parse_args()

    if not SCAPY_AVAILABLE:
        print("Scapy is not available. Install scapy via: pip install scapy")
        sys.exit(1)

    pcap_path = Path(args.pcap)
    if not pcap_path.exists():
        print(f"Error: PCAP file not found at {pcap_path}")
        sys.exit(1)

    checkpoint_path = args.checkpoint or os.path.join("results", args.exp, "global_model.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    # Load PyTorch model
    meta_path = os.path.join("data", "processed", "synthetic", "meta.npz")
    num_classes = 3
    if os.path.exists(meta_path):
        try:
            m = np.load(meta_path, allow_pickle=True)
            if "class_names" in m:
                num_classes = len(m["class_names"])
        except Exception:
            pass

    model = SimpleNet(input_dim=20, num_classes=num_classes)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"\n{BOLD}======================================================{RESET}")
    print(f"{BOLD}   Live PCAP Replay & Federated IDS Telemetry Stream   {RESET}")
    print(f"{BOLD}======================================================{RESET}")
    print(f"Replaying capture: {pcap_path}")
    print(f"Model Checkpoint:  {checkpoint_path}\n")

    packets = rdpcap(str(pcap_path))
    print(f"Loaded {len(packets)} packets. Beginning real-time feature extraction & classification...\n")

    prev_time = None
    alerts_count = 0

    with torch.no_grad():
        for i, pkt in enumerate(packets):
            feats, prev_time = extract_features_from_packet(pkt, prev_time)
            x_tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
            pred = int(probs.argmax())
            confidence = probs[pred]

            time.sleep(args.delay)

            # Format packet summary
            summary = pkt.summary() if hasattr(pkt, 'summary') else f"Packet #{i+1}"
            if len(summary) > 40:
                summary = summary[:37] + "..."

            if pred == 0:
                verdict = f"{GREEN}{BOLD}[NORMAL TRAFFIC]{RESET}"
            else:
                verdict = f"{RED}{BOLD}[IDS ALERT: THREAT CLASS {pred}]{RESET}"
                alerts_count += 1

            print(f"[{i+1:02d}/{len(packets):02d}] {summary:<40} -> {verdict:<35} (Conf: {confidence:.1%})")

    print(f"\n{BOLD}------------------------------------------------------{RESET}")
    print(f"{BOLD}Replay complete: Processed {len(packets)} packets | Alerts triggered: {alerts_count}{RESET}")
    print(f"{BOLD}------------------------------------------------------{RESET}\n")


if __name__ == "__main__":
    main()
