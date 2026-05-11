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

# Find and patch any supervisor config that mentions tensorboard, preventing restart
while IFS= read -r -d '' conf_file; do
    if grep -qi "tensorboard" "$conf_file"; then
        sed -i 's/^\(\s*autostart\s*=\s*\)true/\1false/I' "$conf_file"
        sed -i 's/^\(\s*autorestart\s*=\s*\)\(true\|unexpected\)/\1false/I' "$conf_file"
        echo "=== [provision] Patched $conf_file ==="
    fi
done < <(find /etc/supervisor /etc/supervisord.d -type f \( -name "*.conf" -o -name "*.ini" \) -print0 2>/dev/null)

supervisorctl reread 2>/dev/null || true
supervisorctl update 2>/dev/null || true

# ── Clone repository ─────────────────────────────────────────────────────────
echo "=== [provision] Cloning repository ==="
REPO_URL="https://github.com/PLACEHOLDER/PLACEHOLDER_REPO.git"
TARGET_DIR="/root/llm_power_experiments"

if [ ! -d "$TARGET_DIR" ]; then
    git clone "$REPO_URL" "$TARGET_DIR"
else
    echo "=== [provision] $TARGET_DIR already exists, skipping clone ==="
fi

# ── Install Python dependencies ───────────────────────────────────────────────
echo "=== [provision] Installing Python dependencies ==="
cd "$TARGET_DIR"
uv sync
uv add vllm huggingface datasets pandas

echo "=== [provision] Done ==="
