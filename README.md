# BitNet-MLX

**High-performance MLX-native ternary (1.58-bit) and binary quantization for Apple Silicon.**

A clean, extensible implementation focused on **AbsMean scaling**, drop-in `DynamicBitLinear` layers, and end-to-end Hugging Face → MLX conversion pipelines. Optimized for AMX hardware acceleration.

## Goals
- Maximum efficiency on Apple Silicon (memory density + AMX speed)
- Full transparency — no black-box components
- Production-ready sub-2-bit inference for LLMs + VLMs
- Community-driven open source core

## Features
- AbsMean Ternary Quantization with per-row γ scaling
- Hybrid precision (a8w1.58 / a4w1.58)
- Ternary Vision Encoders & Multimodal Projectors
- Drop-in `DynamicBitLinear` replacement for `nn.Linear`
- Thermal-aware activation routing
- Zero SVD — pure O(N) operations

## Quick Start
```bash
git clone https://github.com/cloudcover95/BitNet-mlx.git
cd BitNet-mlx
python3 -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

bitnet-convert --input microsoft/Phi-3-mini-4k-instruct --output ./assets/ternary_phi3
bitnet-eval
