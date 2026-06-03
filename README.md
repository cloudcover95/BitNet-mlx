# BitNet-mlx

**MLX-native 1.58-bit Ternary Engine + Combinatorial TDA for Sovereign Edge AI**

BitNet-mlx is the **core mathematical engine** of the JuniorCloud LLC sovereign edge technology stack. It provides efficient 1.58-bit ternary quantization (`W ∈ −1, 0, 1`) and combinatorial Topological Data Analysis (TDA) directly in discrete latent space.

## Key Features

- **Ternary Projection Layer** — Projects high-dimensional telemetry, spatial, and quant data into discrete ternary embeddings
- **Combinatorial TDA** — Builds simplicial complexes and computes persistence using only ternary dot products (no floating-point SVD)
- **Full End-to-End Pipeline** — `TernaryPipeline` combining projection, analysis, signature generation, and Parquet persistence
- **Easy Data Ingestion** — `TernaryTelemetry` adapter for lists, numpy arrays, and dictionaries
- **Production Interfaces** — Clean classes for real-world use (`TernaryAnalyzer`, `TernaryPipeline`, `SignatureStore`)
- **LLM Integration Ready** — Designed to work alongside local LLMs (Ollama) via smart routing in JuniorHome

## Why Ternary Quantization?

Ternary weights (`{-1, 0, 1}`) offer an excellent balance between:
- Extreme efficiency on edge hardware (Apple Silicon, future NVIDIA edge chips)
- Good accuracy retention compared to binary quantization
- Natural support for discrete mathematical operations (ideal for TDA and state-space reasoning)

This makes BitNet-mlx particularly powerful for long-running, always-on, sovereign systems where determinism and mathematical structure matter.

## Integration in the Ecosystem

| Repository          | How it uses BitNet-mlx                              |
|---------------------|------------------------------------------------------|
| **JuniorHome**      | Central orchestrator with smart LLM + ternary routing |
| **crispy-mouse**    | Projects kinematic/sensor telemetry into ternary space |
| **JuniorClimbs**    | Performance analysis and movement data reasoning     |
| **JuniorOmega**     | Spatial sensing data processing                      |
| **stocksnode**      | Quant state matrices and predictive modeling         |
| **JuniorMemSys**    | Stores ternary signatures and persistence landscapes |

## Positioning

Unlike cloud-heavy hybrid pipelines that rely on expensive OCR + frontier LLMs for everything, BitNet-mlx focuses on **efficient, local, mathematically structured analysis**. It excels at telemetry, sensor data, state machines, and topological reasoning — while still being able to work alongside general-purpose local LLMs (via JuniorHome’s SmartLLMRouter).

## Getting Started

```bash
pip install -e .
```

See `src/inference/` for the main modules:
- `ternary_projection.py`
- `ternary_tda.py`
- `ternary_analyzer.py`
- `ternary_pipeline.py`
- `ternary_signature.py`
- `signature_store.py`

## Philosophy

Build efficient, sovereign, mathematically grounded intelligence that runs locally and respects user data and hardware constraints.

Part of the JuniorCloud LLC ecosystem — building toward the first BitNet OS for next-generation edge AI hardware.