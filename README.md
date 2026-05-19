# BitNet-MLX: Sub-2-Bit Quantization Matrix

**A Google Gemini x JuniorCloud LLC Sovereign Engine**

BitNet-MLX is a standalone, logic-dense inference and quantization compiler tailored strictly for Apple Silicon (MLX). It serves as the primary sub-2-bit mathematical engine powering the JuniorAGI-SDK stack, enabling zero-trust, off-grid Edge AGI execution.

## I. Architectural Capabilities

The substrate hard-forks dense FP16/BF16 linear layers, dynamically projecting them into ternary ($b1.58$: $\{-1, 0, +1\}$) or pure binary topologies.

### 1. Hardware-Fused Quantization Pipeline
* **AbsMean Ternary Collapse**: Computes $\gamma = \frac{1}{N}\sum|W_{row}|$. Weights are strictly bounded via $W_{q} = \text{round}(\text{clip}(W/\gamma, -1, 1))$.
* **AMX Execution**: Matrix multiplications bypass standard FP16 execution pipelines, utilizing highly optimized MLX `int2` + scale accumulator sequences directly within SRAM.
* **Lossless Precision Routing**: Allows hybrid topologies (e.g., Ternary weights + INT8/INT4 dynamic activations) with sparsification algorithms for pathological outlier preservation.

### 2. Python LLM Interpreter & Integration
* **Zero-AST Mutation**: Provides a drop-in Python API replicating standard HF/MLX-LM syntax, allowing dynamic loading without dangerous `setattr` runtime hacks.
* **PTQ & QAT Bridge**: Seamless pipeline from Hugging Face $\rightarrow$ Post-Training Quantization $\rightarrow$ MLX Compilation. Capable of ingesting Spectral QAT masters generated from the parent JuniorAGI nodes.

## II. System Integration

**Installation:**
```bash
git clone [https://github.com/cloudcover95/BitNet-mlx.git](https://github.com/cloudcover95/BitNet-mlx.git)
cd BitNet-mlx
pip install -e .
```

**CLI Ingestion Pipeline:**
```bash
bitnet-convert --input microsoft/Phi-3-mini-4k-instruct --output ./assets/ternary_phi3
```

**Python Programmatic Implementation:**
```python
import mlx.core as mx
from bitnet_mlx.bitnet_layers import DynamicBitLinear

# Instantiates a natively compiled ternary linear transformation block
layer = DynamicBitLinear(in_d=4096, out_d=4096)
mx.eval(layer.parameters())
```
