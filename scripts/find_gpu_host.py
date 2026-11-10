#!/usr/bin/env python3
"""Scan the local LAN for hosts that might be the Aura GPU machine.

Run this from a machine on the same network as the Windows RTX box (not from
a sandboxed cloud agent, which usually only sees its own VPC subnet).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ipaddress
import platform
import socket
import subprocess
from typing import Iterable, List, Tuple


def local_ipv4s() -> List[str]:
    addresses: List[str] = []
    hostname = socket.gethostname()
    try:
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            addresses.append(info[4][0])
    except socket.gaierror:
        pass

    # UDP trick for the outbound interface address.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        addresses.append(sock.getsockname()[0])
    finally:
        sock.close()
    return sorted(set(addresses))


def candidate_networks(explicit: str | None) -> List[ipaddress.IPv4Network]:
    if explicit:
        return [ipaddress.ip_network(explicit, strict=False)]

    nets: List[ipaddress.IPv4Network] = []
    for addr in local_ipv4s():
        if addr.startswith("127."):
            continue
        nets.append(ipaddress.ip_network(f"{addr}/24", strict=False))
    return nets


def ping_host(ip: str, timeout_s: float = 1.0) -> bool:
    system = platform.system().lower()
    if system == "windows":
        cmd = ["ping", "-n", "1", "-w", str(int(timeout_s * 1000)), ip]
    else:
        cmd = ["ping", "-c", "1", "-W", str(max(1, int(timeout_s))), ip]
    result = subprocess.run(cmd, capture_output=True, check=False)
    return result.returncode == 0


def probe_ports(ip: str, ports: Iterable[int], timeout_s: float = 0.4) -> List[int]:
    open_ports: List[int] = []
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout_s)
        try:
            if sock.connect_ex((ip, port)) == 0:
                open_ports.append(port)
        finally:
            sock.close()
    return open_ports


def scan_network(
    network: ipaddress.IPv4Network,
    ports: List[int],
) -> List[Tuple[str, List[int]]]:
    hosts = [str(ip) for ip in network.hosts()]
    found: List[Tuple[str, List[int]]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pool:
        ping_results = list(pool.map(ping_host, hosts))
        alive = [ip for ip, ok in zip(hosts, ping_results) if ok]
        port_results = list(pool.map(lambda ip: (ip, probe_ports(ip, ports)), alive))

    for ip, open_ports in port_results:
        found.append((ip, open_ports))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network",
        help="CIDR to scan, e.g. 192.168.1.0/24 (default: local /24s)",
    )
    parser.add_argument(
        "--ports",
        default="3030,22,3389,445,8000,8080",
        help="Comma-separated TCP ports to probe on live hosts",
    )
    args = parser.parse_args()
    ports = [int(p.strip()) for p in args.ports.split(",") if p.strip()]

    print(f"Local IPv4 addresses: {', '.join(local_ipv4s()) or '(none)'}")
    networks = candidate_networks(args.network)
    if not networks:
        raise SystemExit("No candidate networks found. Pass --network explicitly.")

    for network in networks:
        print(f"\nScanning {network} ...")
        for ip, open_ports in scan_network(network, ports):
            port_txt = ", ".join(str(p) for p in open_ports) or "no probed ports open"
            print(f"  {ip}: {port_txt}")


if __name__ == "__main__":
    main()
