#!/usr/bin/env zsh
# path: scripts/validate_blackbox.sh
set -euo pipefail

echo "[*] Initiating BitNet-MLX Air-Gap Security Audit..."

# 1. Weight Leakage Detection
LEAKED_WEIGHTS=$(find . -type f \( -name "*.safetensors" -o -name "*.gguf" -o -name "model.bin" \) -not -path "*/\.*" 2>/dev/null || true)
if [ -n "$LEAKED_WEIGHTS" ]; then
    echo "[!] CRITICAL: Model weights detected outside quarantine zone."
    echo "$LEAKED_WEIGHTS"
    exit 1
fi

# 2. Syntax Integrity
python3 -m compileall -q src/ run_offline_blackbox.py

echo "[SUCCESS] Blackbox repository verified. Zero weight leakage detected."
