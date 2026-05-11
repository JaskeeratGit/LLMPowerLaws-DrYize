import json
import subprocess
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "gpuConfig.json"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def get_gpu_ids() -> list[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True, text=True, check=True
    )
    return [int(line.strip()) for line in result.stdout.strip().splitlines()]


def apply_clocks(gpu_id: int, config: dict):
    if config.get("lock_sm_clock") and "sm_clock_mhz" in config:
        freq = config["sm_clock_mhz"]
        subprocess.run(
            ["nvidia-smi", "-i", str(gpu_id), f"--lock-gpu-clocks={freq},{freq}"],
            check=True
        )
        print(f"GPU {gpu_id}: SM clock locked to {freq} MHz")

    if config.get("lock_memory_clock") and "memory_clock_mhz" in config:
        freq = config["memory_clock_mhz"]
        subprocess.run(
            ["nvidia-smi", "-i", str(gpu_id), f"--lock-memory-clocks={freq},{freq}"],
            check=True
        )
        print(f"GPU {gpu_id}: memory clock locked to {freq} MHz")


def verify_clocks_dcgm():
    # SM clock = field 100, memory clock = field 101
    result = subprocess.run(
        ["dcgmi", "dmon", "-e", "100,101", "-c", "1"],
        capture_output=True, text=True, timeout=15
    )
    print("\n--- DCGM clock verification ---")
    print(result.stdout if result.stdout else result.stderr)


def verify_clocks_nvsmi():
    result = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=index,clocks.sm,clocks.mem",
         "--format=csv,noheader"],
        capture_output=True, text=True, check=True
    )
    print("\n--- nvidia-smi clock verification (index, SM MHz, mem MHz) ---")
    print(result.stdout.strip())


def verify_clocks():
    try:
        verify_clocks_dcgm()
    except FileNotFoundError:
        # dcgmi not present, fall back to nvidia-smi
        verify_clocks_nvsmi()


def main():
    config = load_config()
    gpu_ids = get_gpu_ids()
    print(f"Found {len(gpu_ids)} GPU(s): {gpu_ids}")

    for gpu_id in gpu_ids:
        apply_clocks(gpu_id, config)

    verify_clocks()


if __name__ == "__main__":
    main()
