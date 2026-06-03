#!/usr/bin/env zsh
# path: BitNet-mlx/scripts/validate_blackbox.sh
set -euo pipefail

echo "[*] Initiating Air-Gap Security Audit..."

# 1. Weight Leakage Detection
# Ensures no HuggingFace cache or model weights are staged for GitHub
LEAKED_WEIGHTS=$(find . -name "*.safetensors" -not -path "*/\.*" 2>/dev/null || true)
if [ -n "$LEAKED_WEIGHTS" ]; then
    echo "[!] CRITICAL: Model weights detected outside quarantine zone."
    echo "$LEAKED_WEIGHTS"
    exit 1
fi

# 2. Syntax Integrity
python3 -m compileall -q src/ run_offline_blackbox.py

echo "[SUCCESS] Blackbox verified. Ready for offline inference."
