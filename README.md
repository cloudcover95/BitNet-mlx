# BitNet-MLX

**High-performance MLX-native ternary (1.58-bit) and binary quantization for Apple Silicon.**

Engineered by JuniorCloud LLC. Focused on **AbsMean scaling**, drop-in `DynamicBitLinear` layers, and end-to-end Hugging Face → MLX conversion pipelines. Fully optimized for local AMX execution without vendor lock-in.

## Phase 2: VLM Sovereign Integration
Natively bridges multi-modality targets (`TernaryVisionEncoder`, `VLMCompatibilityLayer`) while executing $O(N)$ ternary KV-cache compression layers directly within Apple Silicon's unified memory registers.
