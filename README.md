# BitNet-MLX: Enterprise Quantization Compiler

**Sovereign Edge-Native Matrix Engine for Apple Silicon**

BitNet-MLX is a high-performance, standalone quantization compiler powering the JuniorAGI-SDK stack. Engineered by JuniorCloud LLC, it mathematically collapses dense FP16/BF16 tensor manifolds into extreme low-bit topologies (Ternary $b1.58$: $\{-1, 0, 1\}$ and Binary: $\{-1, 1\}$), executing directly on Apple's Matrix Coprocessor (AMX).

This substrate ensures absolute operational sovereignty and maximum thermodynamics-to-compute (C2V) efficiency for edge nodes.

## Core Compiler Capabilities

### 1. Ternary & Binary Execution Engine
* **AbsMean Quantization Pipeline**: Implements native $b1.58$-style scaling via average absolute value metrics ($\gamma = \frac{1}{N}\sum|W_{row}|$), mathematically bounding operations without eroding topological integrity.
* **Lossless Inference Modularity**: Supports hybrid precision manifolds—ternary weights mapped against INT8 activations (`a8w1.58` / `a4w1.58`) with sparsification routines for systemic outlier handling.
* **Spectral Rank Preservation**: Facilitates SVD-regularized ($A = U \Sigma V^T$) matrices generated via QAT workflows, preventing dimensional collapse during bit-width reduction.

### 2. MLX-Optimized UMA Integration
* **SRAM-Fused Kernels**: Matrix operations execute custom ternary kernels, replacing standard multiplications with hyper-fast AMX additions and lookups.
* **Memory Density Shift**: Reduces static VRAM footprint by up to 7x compared to standard FP16 topologies, allowing 100B-class models to execute flawlessly within consumer-grade Unified Memory Architectures (UMA) such as the M4 Max / M1 Ultra chips.
* **Deterministic Autoregression**: Supports high-throughput, token-by-token streaming with zero internal state mutation (no runtime `setattr` injection), ensuring absolute KV-cache safety.

### 3. Substrate Python API
* **Drop-In Linear Replacements**: Exposes `DynamicBitLinear` for programmatic integration into arbitrary transformer topologies.
* **Hugging Face / MLX Bridge**: End-to-end ingestion pipeline: `HF Remote -> AbsMean Quantization -> MLX Compile -> AMX Execution`.

## Integration Protocol

**Environment Initialization:**
```bash
git clone [https://github.com/cloudcover95/BitNet-mlx.git](https://github.com/cloudcover95/BitNet-mlx.git)
cd BitNet-mlx
pip install -e .
```

**CLI Execution:**
```bash
bitnet-convert --input microsoft/Phi-3-mini-4k-instruct --output ./assets/ternary_phi3
```

**Programmatic Invocation:**
```python
import mlx.core as mx
from bitnet_mlx.bitnet_layers import DynamicBitLinear

# Instantiate hardware-fused ternary linear transformation block
layer = DynamicBitLinear(in_d=4096, out_d=4096)
mx.eval(layer.parameters())
```
