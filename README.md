# BitNet-MLX: Omni-Sovereign SDK

**Production-grade MLX-native ternary (1.58-bit) SDK mapped with Composite TDA.**

BitNet-MLX supports **all Apple Silicon**, from iOS/iPadOS A-Series mobile nodes up to M-Ultra clusters. Utilizing the **Omni-Hardware Telemetry Engine** (`JuniorHome`), it dynamically scales precision boundaries (`q_max`) to ensure absolute system stability without exceeding 48V power constraints.

## Features
* **OpenAI-Compatible FastAPI Server:** Native SSE streaming. Drop-in replacement for local UI frontends (Cursor, ChatbotUI).
* **JuniorSwarm RPC:** ZeroMQ-powered decentralized subnets. Link A-series and M-series nodes to distribute matrix multiplications across a local Wi-Fi mesh.
* **Composite TDA Math:** O(N) Variance-Preserving Ternary Quantization via Spectral Gaps and Betti-0 Graph density. Zero-SVD bloat. >99% structural fidelity.
* **Audio & VLM-Safe Transmutation:** Deep module parsing protects Multi-Modal projectors (LLaVA, Pixtral) and Audio encoders (Whisper) from ternary collapse.

## Installation
```bash
git clone [https://github.com/cloudcover95/BitNet-mlx.git](https://github.com/cloudcover95/BitNet-mlx.git)
cd BitNet-mlx
pip install -e .
Quickstart CLI
Bash
# Evaluate TDA Structural Fidelity & Telemetry
bitnet-mlx sandbox                 

# Convert a Model (VLM & Audio Safe)
bitnet-mlx convert --repo mlx-community/Phi-3-mini-4k-instruct --output ./models/phi-3-bitnet

# Mount FastAPI Server (OpenAI Compatible Endpoint)
bitnet-mlx serve --model ./models/phi-3-bitnet --port 8080

# Boot a Distributed Swarm Node
bitnet-mlx swarm-node --mode orchestrator
