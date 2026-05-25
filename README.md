# BitNet-MLX: JuniorCloud TDA Matrix

**Production-grade MLX-native ternary (1.58-bit) SDK for Apple Silicon.**

## Core Engines
- **Topology Surgeon (`converter.py`):** Transmutes HuggingFace/MLX-LM FP16 models into zero-SVD 1.58-bit state boundaries.
- **Injection Engine (`inference.py`):** Autonomous, edge-native inference pipeline capable of dynamically mounting transformed weights.
- **JuniorCloud TDA Evaluator (`evaluation.py`):** Adaptive scaling telemetry guaranteeing >99% variance preservation on quantized matrices.

## Execution Matrix
```bash
# Verify SDK integrity (>99% Variance Recovery Target)
bitnet-mlx eval

# Execute Converter Pipeline
bitnet-mlx convert --repo mlx-community/Phi-3-mini-4k-instruct --output ./models/phi-3-bitnet

# Autonomous Edge Inference
bitnet-mlx chat --model ./models/phi-3-bitnet --prompt "Detail the topology of ternary mapping."
