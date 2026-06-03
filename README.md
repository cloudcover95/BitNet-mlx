# BitNet-mlx

**MLX-native 1.58-bit Ternary Engine + Combinatorial TDA**

BitNet-mlx is the core mathematical engine for the JuniorCloud LLC sovereign edge stack. It provides efficient 1.58-bit ternary projection and combinatorial Topological Data Analysis (TDA) entirely in discrete latent space.

## Current Capabilities

- **Ternary Projection** — High-dimensional data (telemetry, spatial, quant) projected into `W ∈ −1, 0, 1` space
- **Combinatorial TDA** — Simplicial complex construction and persistence using only ternary dot products
- **Full Pipeline** — End-to-end `TernaryPipeline` with projection + TDA + structured signatures
- **Parquet Persistence** — Clean `TernarySignature` + `SignatureStore` ready for JuniorMemSys
- **Easy Ingestion** — `TernaryTelemetry` adapter for lists, numpy, and dicts
- **Production Interfaces** — `TernaryAnalyzer` and `TernaryPipeline` for real use

## Philosophy

We do **not** rely on continuous SVD or heavy floating-point dimensionality reduction on edge. Instead, we project into discrete ternary space early and perform topology directly there. This is faster, more private, and more efficient for long-running sovereign systems.

## Integration

BitNet-mlx is designed to be consumed by:
- **JuniorHome** (central orchestrator / future BitNet OS)
- **crispy-mouse** (via TernaryTelemetryAdapter)
- **JuniorOmega** and **JuniorClimbs** (spatial/performance data)
- **stocksnode** (quant state matrices)

## Positioning vs Cloud Hybrids

Unlike hybrid cloud pipelines that fall back to expensive OCR + LLM for everything, BitNet-mlx focuses on efficient, deterministic, local-first analysis in ternary space. It excels at structured data, telemetry, and mathematical state — not generic document OCR.

Part of building a complete sovereign edge technology stack under JuniorCloud LLC.