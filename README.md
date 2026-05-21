# BitNet-MLX

**High-performance MLX-native ternary (1.58-bit) quantization for Apple Silicon.**

Focused on **AbsMean scaling**, drop-in `DynamicBitLinear` layers, and Omni-modality execution (Text, Vision, Audio). Fully optimized for local AMX execution without vendor lock-in, enabling sovereign edge-native clusters.

## Phase 3: Omni-Swarm Integration
Natively bridges Audio topologies (`TernaryAudioEncoder`) and multi-node tensor sharding via `Binary RPC Swarm`. Dynamic $\gamma$ scaling allows the engine to autonomously adjust AbsMean mapping based on the ingested modality, maintaining strict $O(N)$ efficiency.
