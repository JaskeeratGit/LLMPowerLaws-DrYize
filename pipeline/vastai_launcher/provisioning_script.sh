#!/bin/bash
set -euo pipefail

echo "=== [provision] Starting ==="

# ── Install DCGM ─────────────────────────────────────────────────────────────
echo "=== [provision] Installing DCGM ==="
apt-get update -y
apt-get install -y datacenter-gpu-manager

# ── Disable tensorboard (supervisorctl) ──────────────────────────────────────
echo "=== [provision] Disabling tensorboard in supervisord ==="
supervisorctl stop tensorboard 2>/dev/null || true


echo "=== [provision] Done ==="
