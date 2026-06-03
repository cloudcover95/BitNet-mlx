# BitNet-mlx

**MLX-native 1.58-bit Ternary Quantization + Proprietary Blackbox (V1.1)**

## V1.1 Production Hardening
- Hardened `pyproject.toml` with hatchling packaging
- Strict `.gitignore` for model weight quarantine
- `Makefile` for consistent developer workflow
- `tests/test_blackbox.py` for math kernel integrity
- `scripts/validate_blackbox.sh` for air-gap security audit

This repository is now a fully isolated, sovereign, offline inference engine.

## Quick Start

```bash
make install
make test
make run
```