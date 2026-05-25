# BitNet-MLX: Absolute-Zero Engine

**Production-grade MLX-native ternary (1.58-bit) SDK for Apple Silicon.**

## Core Engines
- **Topology Surgeon (`converter.py`):** Transmutes HuggingFace/MLX-LM FP16 models into zero-SVD 1.58-bit state boundaries.
- **Injection Engine (`inference.py`):** Autonomous, edge-native inference pipeline capable of dynamically mounting transformed weights.
- **Grok Evaluator (`evaluation.py`):** Real-time parametric drift tracking saved immutably to `.parquet`, protected against FP16 accumulation overflow.

## Execution Matrix
```bash
bitnet-mlx eval
bitnet-mlx convert --repo mlx-community/Phi-3-mini-4k-instruct --output ./models/phi-3-bitnet
bitnet-mlx chat --model ./models/phi-3-bitnet --prompt "Detail the topology of ternary mapping."
