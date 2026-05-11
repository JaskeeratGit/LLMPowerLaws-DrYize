#!/usr/bin/env python3
"""
Loads all prompts from dataset_loader, sends them to the vLLM server at
max_concurrent_requests concurrency, and monitors GPU power draw at 100 Hz
via DCGM for the full duration of inference.

Results are saved to inference_runner/results/<model>_<timestamp>/
  power_log.csv   - timestamped DCGM power readings (100 Hz)
  results.jsonl   - per-request metrics:
      request_duration_s  total wall time for the request
      ttft_s              time to first token
      tpot_s              time per output token (generation latency / output_tokens)
      tokens_per_second   output_tokens / request_duration
      prompt_tokens       input length in tokens
      completion_tokens   output length in tokens
      error               null on success, error string on failure
"""

import asyncio
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import aiohttp

VLLM_CONFIG_PATH = Path(__file__).parent.parent / "vllm_server_launcher" / "vllmConfig.json"
PROMPTS_PATH = Path(__file__).parent.parent / "dataset_loader" / "prompts.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ── DCGM monitoring ───────────────────────────────────────────────────────────

def start_dcgm_monitor(log_path: Path) -> tuple:
    """Launch dcgmi dmon at 100 Hz; writes timestamped lines to log_path."""
    proc = subprocess.Popen(
        ["dcgmi", "dmon", "-e", "155", "-d", "10"],  # field 155 = power (W), 10ms = 100 Hz
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    stop_event = threading.Event()
    log_file = open(log_path, "w")
    log_file.write("timestamp,raw\n")
    log_file.flush()

    def _reader():
        for line in proc.stdout:
            if stop_event.is_set():
                break
            log_file.write(f"{time.time():.6f},{line.rstrip()}\n")
            log_file.flush()
        log_file.close()

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    return proc, thread, stop_event


def stop_dcgm_monitor(proc, thread, stop_event):
    stop_event.set()
    proc.terminate()
    proc.wait(timeout=5)
    thread.join(timeout=5)


# ── vLLM health check ─────────────────────────────────────────────────────────

async def wait_for_server(base_url: str, timeout: int = 60):
    health_url = f"{base_url}/health"
    deadline = time.time() + timeout
    async with aiohttp.ClientSession() as session:
        while time.time() < deadline:
            try:
                async with session.get(health_url) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                pass
            await asyncio.sleep(2)
    sys.exit(f"Error: vLLM server at {base_url} did not become ready within {timeout}s.")


# ── Inference ─────────────────────────────────────────────────────────────────

async def send_request(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url: str,
    model: str,
    prompt: str,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    async with semaphore:
        t_start = time.perf_counter()
        t_first_token = None
        prompt_tokens = None
        completion_tokens = None
        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                async for raw_line in resp.content:
                    line = raw_line.decode().strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str == "[DONE]":
                        break
                    chunk = json.loads(data_str)

                    # Record wall time of the first content token
                    if t_first_token is None:
                        choices = chunk.get("choices", [])
                        if choices and choices[0].get("delta", {}).get("content"):
                            t_first_token = time.perf_counter()

                    # Usage arrives in the final chunk (stream_options.include_usage)
                    usage = chunk.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens")
                        completion_tokens = usage.get("completion_tokens")

            t_end = time.perf_counter()
            duration = t_end - t_start
            ttft = (t_first_token - t_start) if t_first_token is not None else None
            generation_time = (t_end - t_first_token) if t_first_token is not None else None
            tpot = (generation_time / completion_tokens) if (generation_time and completion_tokens) else None
            tps = (completion_tokens / duration) if (completion_tokens and duration) else None

            return {
                "request_duration_s": round(duration, 6),
                "ttft_s": round(ttft, 6) if ttft is not None else None,
                "tpot_s": round(tpot, 6) if tpot is not None else None,
                "tokens_per_second": round(tps, 3) if tps is not None else None,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "error": None,
            }
        except Exception as e:
            return {
                "request_duration_s": round(time.perf_counter() - t_start, 6),
                "ttft_s": None, "tpot_s": None, "tokens_per_second": None,
                "prompt_tokens": None, "completion_tokens": None,
                "error": str(e),
            }


async def run_inference(prompts: list[str], config: dict, results_path: Path):
    base_url = "http://localhost:8000"
    completions_url = f"{base_url}/v1/chat/completions"
    model = config["model"]
    concurrency = config["max_concurrent_requests"]

    print(f"Waiting for vLLM server at {base_url}...")
    await wait_for_server(base_url)
    print("Server is ready.")

    semaphore = asyncio.Semaphore(concurrency)
    total = len(prompts)
    print(f"Sending {total} requests (concurrency={concurrency})...")

    completed = 0

    async with aiohttp.ClientSession() as session:
        tasks = [
            send_request(session, semaphore, completions_url, model, p)
            for p in prompts
        ]
        with open(results_path, "w") as out:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                out.write(json.dumps(result) + "\n")
                completed += 1
                if completed % 1000 == 0 or completed == total:
                    print(f"  {completed}/{total} requests completed")

    lines = [json.loads(line) for line in results_path.read_text().splitlines()]
    errors = sum(1 for r in lines if r["error"])
    print(f"Done. {total - errors}/{total} succeeded. Results saved to {results_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def load_prompts_from_file(path: Path) -> list[str]:
    if not path.exists():
        sys.exit(
            f"Error: {path} not found.\n"
            "Run dataset_loader first: python pipeline/dataset_loader/dataset_loader.py"
        )
    with open(path) as f:
        return [json.loads(line)["prompt"] for line in f if line.strip()]


def main():
    vllm_config = load_json(VLLM_CONFIG_PATH)

    print(f"Loading prompts from {PROMPTS_PATH}...")
    prompts = load_prompts_from_file(PROMPTS_PATH)
    print(f"Total prompts: {len(prompts)}\n")

    model_slug = vllm_config["model"].replace("/", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"{model_slug}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    power_log_path = run_dir / "power_log.csv"
    results_path = run_dir / "results.jsonl"

    print("Starting DCGM power monitoring at 100 Hz...")
    try:
        dcgm_proc, dcgm_thread, stop_event = start_dcgm_monitor(power_log_path)
    except FileNotFoundError:
        sys.exit("Error: 'dcgmi' not found. Ensure DCGM is installed and in PATH.")

    try:
        asyncio.run(run_inference(prompts, vllm_config, results_path))
    finally:
        print("Stopping DCGM monitor...")
        stop_dcgm_monitor(dcgm_proc, dcgm_thread, stop_event)
        print(f"Power log saved to {power_log_path}")


if __name__ == "__main__":
    main()
