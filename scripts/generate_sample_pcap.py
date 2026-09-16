#!/usr/bin/env python
"""
generate_sample_pcap.py

Generates a synthetic PCAP packet capture file (`data/sample_traffic.pcap`)
containing realistic normal IoT traffic and simulated attack packets
(DDoS SYN flood, Port Scanning, Injection payloads) using Scapy.

USAGE
-----
    python scripts/generate_sample_pcap.py --output data/sample_traffic.pcap --count 30
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scapy.all import Ether, IP, TCP, UDP, ICMP, Raw, wrpcap
except ImportError:
    print("Scapy is required. Run: pip install scapy")
    sys.exit(1)


def build_sample_packets(num_packets=30, seed=42):
    random.seed(seed)
    packets = []

    gateway_ip = "192.168.1.1"
    iot_devices = ["192.168.1.10", "192.168.1.11", "192.168.1.12", "192.168.1.15"]
    external_attacker = "45.33.32.156"

    current_time = time.time() - 60  # start 60s in past

    for i in range(num_packets):
        # Add realistic inter-packet timestamp offset (0.01s to 0.3s)
        current_time += random.uniform(0.01, 0.3)

        # Classify sequence pattern: ~40% Normal, ~30% DDoS, ~20% Scan, ~10% Injection
        r = random.random()

        if r < 0.4:
            # NORMAL TRAFFIC: MQTT / HTTP telemetry
            src_ip = random.choice(iot_devices)
            dst_ip = gateway_ip
            proto_type = random.choice(["MQTT", "HTTP"])
            if proto_type == "MQTT":
                sport = random.randint(49152, 65535)
                dport = 1883
                payload = b"\x10\x1a\x00\x04MQTT\x04\x02\x00\x3c\x00\x0eIoT_Sensor_Node"
                pkt = Ether() / IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=dport, flags="PA") / Raw(load=payload)
            else:
                sport = random.randint(49152, 65535)
                dport = 80
                payload = b"GET /api/v1/telemetry HTTP/1.1\r\nHost: 192.168.1.1\r\nUser-Agent: IoT-Sensor/1.0\r\n\r\n"
                pkt = Ether() / IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=dport, flags="PA") / Raw(load=payload)

        elif r < 0.7:
            # DDOS ATTACK: High rate SYN flood from external attacker
            src_ip = f"10.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
            dst_ip = "192.168.1.10"
            sport = random.randint(1024, 65535)
            dport = 80
            pkt = Ether() / IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=dport, flags="S", window=64240)

        elif r < 0.9:
            # RECONNAISSANCE / PORT SCAN: Sequential port probing
            src_ip = external_attacker
            dst_ip = "192.168.1.1"
            sport = 54321
            dport = random.choice([21, 22, 23, 80, 443, 8080, 3306, 502, 1883])
            pkt = Ether() / IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=dport, flags="S")

        else:
            # INJECTION / EXPLOIT: HTTP POST with malicious payload
            src_ip = external_attacker
            dst_ip = "192.168.1.10"
            sport = random.randint(49152, 65535)
            dport = 80
            payload = b"POST /login HTTP/1.1\r\nHost: iot-admin\r\n\r\nusername=admin' OR '1'='1&pass=123"
            pkt = Ether() / IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=dport, flags="PA") / Raw(load=payload)

        pkt.time = current_time
        packets.append(pkt)

    return packets


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic sample PCAP file")
    parser.add_argument("--output", default="data/sample_traffic.pcap", help="Output .pcap filepath")
    parser.add_argument("--count", type=int, default=30, help="Number of packets to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} synthetic network packets...")
    packets = build_sample_packets(num_packets=args.count, seed=args.seed)
    wrpcap(str(output_path), packets)

    print(f"Successfully created PCAP capture: {output_path} ({os.path.getsize(output_path)} bytes)")


if __name__ == "__main__":
    main()
