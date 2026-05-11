#!/usr/bin/env python3
"""
Launch a vLLM OpenAI-compatible server from vllmConfig.json inside a
detached tmux session named 'vllm'. Any existing vLLM processes and the
previous tmux session are killed first so only one server runs at a time.

Config fields:
  model                  - HuggingFace model ID or local path
  max_num_seqs           - batch size (max sequences in-flight per iteration)
  tensor_parallel_size   - number of GPUs for tensor parallelism
  kv_cache_dtype         - KV cache quantization: auto | fp8 | fp8_e5m2 | fp8_e4m3
  max_concurrent_requests - max parallel API requests accepted by the server
"""

import json
import subprocess
import sys
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "vllmConfig.json"
TMUX_SESSION = "vllm"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def build_command(config: dict) -> list[str]:
    required = ["model", "max_num_seqs", "tensor_parallel_size", "kv_cache_dtype", "max_concurrent_requests"]
    missing = [k for k in required if k not in config]
    if missing:
        sys.exit(f"Error: missing required config keys: {missing}")

    return [
        "vllm", "serve", config["model"],
        "--max-num-seqs", str(config["max_num_seqs"]),
        "--tensor-parallel-size", str(config["tensor_parallel_size"]),
        "--kv-cache-dtype", config["kv_cache_dtype"],
        "--max-concurrent-requests", str(config["max_concurrent_requests"]),
    ]


def kill_existing_vllm():
    # Kill the named tmux session if it exists
    result = subprocess.run(
        ["tmux", "has-session", "-t", TMUX_SESSION],
        capture_output=True
    )
    if result.returncode == 0:
        subprocess.run(["tmux", "kill-session", "-t", TMUX_SESSION], check=True)
        print(f"Killed existing tmux session '{TMUX_SESSION}'.")

    # Also kill any stray vllm processes not managed by tmux
    result = subprocess.run(
        ["pkill", "-f", "vllm serve"],
        capture_output=True
    )
    if result.returncode == 0:
        print("Killed stray vllm serve process(es).")


def launch_in_tmux(cmd: list[str]):
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", TMUX_SESSION, " ".join(cmd)],
        check=True
    )


def main():
    config = load_config()

    print("vLLM server config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    kill_existing_vllm()

    cmd = build_command(config)
    print(f"Launching in tmux session '{TMUX_SESSION}': {' '.join(cmd)}")

    try:
        launch_in_tmux(cmd)
    except FileNotFoundError:
        sys.exit("Error: 'tmux' not found. Install with: apt install tmux")

    print(f"\nServer is running. To attach: tmux attach -t {TMUX_SESSION}")
    print(f"To stop:             tmux kill-session -t {TMUX_SESSION}")


if __name__ == "__main__":
    main()
