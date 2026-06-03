# BitNet-mlx

**MLX-native 1.58-bit Ternary Quantization + Proprietary Blackbox**

## New in V1.0
- `src/inference/proprietary_blackbox.py` — Full proprietary math kernel (SVD, KPZ, Identity Drift, Q-Mark)
- `src/inference/quantization_calibration.py` — Production MLX quantization calibration (AbsMean + per-channel)
- `run_offline_blackbox.py` — Air-gapped FastAPI server

The engine now supports explicit quantization calibration before inference.