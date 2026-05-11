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

# ── Clone repository ─────────────────────────────────────────────────────────
echo "=== [provision] Cloning repository ==="
REPO_URL="https://github.com/JaskeeratGit/LLMPowerLaws-DrYize.git"
TARGET_DIR="/root/llm_power_experiments"

if [ ! -d "$TARGET_DIR" ]; then
    git clone "$REPO_URL" "$TARGET_DIR"
else
    echo "=== [provision] $TARGET_DIR already exists, skipping clone ==="
fi

echo "=== [provision] Done ==="
