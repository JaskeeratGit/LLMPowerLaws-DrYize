#!/usr/bin/env python3
"""
Vast.ai instance launcher.

Enforces NVIDIA driver >= 575 and uses the NVIDIA CUDA template
(https://cloud.vast.ai/template/readme/9e78853a2a7cf4c576aed9bba21e65db).
On startup the instance runs provisioning_script.sh.
"""

import os
import sys
import json
import time
import base64
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from vastai import VastAI


SCRIPT_DIR = Path(__file__).parent
# NVIDIA CUDA template (vastai/base-image)
TEMPLATE_HASH = "3771d64fe404be8104d8180782435b48"
DRIVER_VERSION_FILTER = "driver_version>=575.0.0"
POLL_INTERVAL_SECONDS = 15
MAX_WAIT_SECONDS = 600


def load_env(env_path: Path) -> None:
    if not env_path.exists():
        sys.exit(f"Error: .env file not found at {env_path}")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def print_offers_table(offers: list) -> None:
    header = f"{'ID':>10}  {'GPU':<22}  {'#GPUs':>5}  {'$/hr':>7}  {'Driver':<14}  {'VRAM(GB)':>8}"
    print(header)
    print("-" * len(header))
    for o in offers:
        vram_gb = o.get("gpu_total_ram", 0) / 1024
        print(
            f"{o['id']:>10}  {o.get('gpu_name', '?'):<22}  "
            f"{o.get('num_gpus', 0):>5}  "
            f"{o.get('dph_total', 0):>7.3f}  "
            f"{str(o.get('driver_version', '?')):<14}  "
            f"{vram_gb:>8.1f}"
        )


def wait_for_running(vast: VastAI, instance_id: int) -> dict:
    start = time.time()
    while True:
        elapsed = int(time.time() - start)
        try:
            instance = vast.show_instance(id=instance_id)
        except Exception as e:
            print(f"  [{elapsed}s] Error polling instance: {e}")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        status = instance.get("actual_status", "unknown")
        print(f"  [{elapsed}s] Status: {status}")

        if status == "running":
            return instance

        if elapsed > MAX_WAIT_SECONDS:
            sys.exit(
                f"Error: instance did not reach 'running' within {MAX_WAIT_SECONDS}s. "
                "Check the vast.ai console."
            )

        time.sleep(POLL_INTERVAL_SECONDS)


def main():
    load_env(SCRIPT_DIR / ".env")
    api_key = os.environ.get("VAST_API_KEY")
    if not api_key:
        sys.exit("Error: VAST_API_KEY not set in .env")

    vast = VastAI(api_key=api_key)

    # ── Step 1: Search for offers ─────────────────────────────────────────────
    print("=" * 62)
    print("Step 1: Search for GPU offers  (driver_version >= 575 enforced)")
    print("=" * 62)
    user_query = input(
        "\nQuery (e.g. gpu_name=RTX_4090 num_gpus=1 reliability>0.99 rentable=true): "
    ).strip()
    limit = input("Number of offers to retrieve [default 5]: ").strip() or "5"

    full_query = f"{user_query} {DRIVER_VERSION_FILTER}".strip()
    print(f"\nSearching: {full_query}\n")

    offers = vast.search_offers(query=full_query, order="dph_total", limit=limit)

    if not offers:
        sys.exit("No offers found. Try relaxing your filters.")

    offers_path = SCRIPT_DIR / "offers.json"
    with open(offers_path, "w") as f:
        json.dump(offers, f, indent=2)

    print(f"Found {len(offers)} offer(s). Full details saved to {offers_path}.\n")
    print_offers_table(offers)

    # ── Step 2: Choose an offer ───────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("Step 2: Choose an offer")
    print("=" * 62)
    offer_id = input("\nOffer ID to launch: ").strip()
    storage = input("Storage in GB [default 50]: ").strip() or "50"


    # # ── Step 3: Build onstart_cmd from provisioning script ────────────────────
    # provisioning_path = SCRIPT_DIR / "provisioning_script.sh"
    # if not provisioning_path.exists():
    #     sys.exit(f"Error: {provisioning_path} not found")

    # encoded_script = base64.b64encode(provisioning_path.read_bytes()).decode()
    # onstart_cmd = f"echo {encoded_script} | base64 -d | bash"

    # ── Step 4: Launch instance ───────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("Step 3: Launching instance")
    print("=" * 62)
    print(f"\nUsing NVIDIA CUDA template: {TEMPLATE_HASH}")

    result = vast.create_instance(
        id=int(offer_id),
        template_hash=TEMPLATE_HASH,
        disk=float(storage),
        runtype="ssh",
    )

    if not result or not isinstance(result, dict):
        sys.exit(f"Error: unexpected response from create_instance: {result}")

    instance_id = result.get("new_contract")
    if not instance_id:
        sys.exit(f"Error: no instance ID in response: {result}")

    print(f"\nInstance created — ID: {instance_id}")

    # ── Step 5: Wait for running ──────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("Step 4: Waiting for instance to start")
    print("=" * 62 + "\n")

    wait_for_running(vast, instance_id)
    print("\nInstance is running!")


    # ── Step 6: SSH ───────────────────────────────────────────────────────────
    ## YOU MUST HAVE AN SSH KEY REGISTERED IN YOUR VAST.AI ACCOUNT FOR THIS.
    print("\n" + "=" * 62)
    print("Step 5: Connecting via SSH")
    print("=" * 62)

    ssh_url = vast.ssh_url(id=instance_id)
    if not ssh_url:
        sys.exit("Error: could not retrieve SSH URL. Check vast.ai console.")

    # sdk returns "ssh://user@host:port"
    parsed = urlparse(ssh_url)
    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-p", str(parsed.port),
        f"{parsed.username}@{parsed.hostname}",
    ]
    print(f"\nSSH command: {' '.join(ssh_cmd)}")
    print("Connecting...\n")
    subprocess.call(ssh_cmd)


if __name__ == "__main__":
    main()
