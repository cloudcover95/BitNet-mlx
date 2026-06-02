import json
import shutil
import glob
import logging
from pathlib import Path
import mlx.nn as nn
import mlx.core as mx
from mlx_lm.utils import load, save_weights
from huggingface_hub import snapshot_download
from ..core.dynamic_bitlinear import DynamicBitLinear

logger = logging.getLogger("TopologySurgeon")

class TopologySurgeon:
    """Safely replaces dense FP16 arrays with DynamicBitLinear, preserving multimodal semantics."""
    
    @staticmethod
    def transmute(module: nn.Module) -> nn.Module:
        # Strict bypass logic for LLM, VLM (LLaVA/Pixtral), and Audio (Whisper) topological bounds
        protected_gates = [
            "lm_head", "embed_tokens", "embed", "gate", 
            "vision_proj", "multi_modal", "image_encoder", 
            "conv", "wte", "wpe", "norm", "ln_",
            "audio_encoder", "feature_extractor", "mel"
        ]
        
        def _replace(mod: nn.Module, prefix: str = "") -> None:
            for name, child in list(mod.named_children()):
                path = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.Linear) and not any(k in path.lower() for k in protected_gates):
                    in_dim = child.weight.shape[1]
                    out_dim = child.weight.shape[0]
                    has_bias = child.bias is not None
                    
                    new_layer = DynamicBitLinear(in_features=in_dim, out_features=out_dim, bias=has_bias)
                    new_layer.weight = child.weight
                    if has_bias:
                        new_layer.bias = child.bias
                    setattr(mod, name, new_layer)
                else:
                    _replace(child, path)
        _replace(module)
        return module

    @staticmethod
    def build_manifold(repo_id: str, output_dir: str) -> None:
        try:
            logger.info(f"Ingesting FP16 topology from {repo_id}...")
            model, tokenizer = load(repo_id)
            
            logger.info("Executing Topological Transmutation (1.58-bit Projection with Multi-Modal Bypass)...")
            quantized_model = TopologySurgeon.transmute(model)
            
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            
            logger.info("Serializing 1.58-bit Safetensors...")
            save_weights(out_path, dict(quantized_model.parameters()))
            
            local_dir = snapshot_download(repo_id, allow_patterns=["*.json", "tokenizer*", "preprocessor*"])
            for file in glob.glob(f"{local_dir}/*"):
                shutil.copy(file, out_path)
            
            config_path = out_path / "config.json"
            if config_path.exists():
                with open(config_path, "r+") as f:
                    config = json.load(f)
                    config["quantization"] = {"group_size": 0, "bits": 2, "type": "bitnet-1.58-omni-tda"}
                    f.seek(0)
                    json.dump(config, f, indent=2)
                    f.truncate()
            logger.info(f"Matrix successfully secured at {output_dir}")
        except Exception as e:
            logger.error(f"Surgical failure: {e}")
            raise