# BitNet-mlx

**MLX-native 1.58-bit Ternary Quantization + Proprietary Blackbox (V1.2)**

## V1.2 Enterprise CI/CD
- GitHub Actions workflow targeting `macos-latest` for native MLX/Metal testing
- Hardened air-gap security auditor
- Deterministic math kernel validation on Apple Silicon

The CI now guarantees that all proprietary SVD/KPZ logic compiles and runs correctly on M-series hardware before merge.

## Quick Start

```bash
make install
make test
make run
```