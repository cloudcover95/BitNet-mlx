#!/usr/bin/env bash
# private_blackbox/emulate.sh
source ../.venv/bin/activate
echo "=========================================="
echo " JUNIORCLOUD MEMSYS EMULATOR (OFF-GRID)   "
echo "=========================================="
python run_tests.py
echo ""
python memsys_engine.py