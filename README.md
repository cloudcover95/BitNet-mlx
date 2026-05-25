# BitNet-MLX: Sovereign Converter Engine

**Production-grade MLX-native ternary (1.58-bit) SDK for Apple Silicon.**

## Architecture Capabilities
- **Topological Converter:** Recursively swaps FP16 models into `DynamicBitLinear` 1.58-bit state boundaries.
- **Venv Enforced Pipeline:** Guarantees isolation from macOS system-level Python constraints.
- **Zero-SVD Assurance:** Operations remain O(N), bound natively to AMX coprocessor grids.

## Execution Routing
```bash
# Evaluate Parametric Weighting
bitnet-mlx eval

# Convert HuggingFace Topologies
bitnet-mlx convert --repo mlx-community/Phi-3-mini-4k-instruct --output ./models/phi-3-bitnet

# Autonomous Edge Execution
bitnet-mlx chat --model ./models/phi-3-bitnet --prompt "Define topological sparsity."
