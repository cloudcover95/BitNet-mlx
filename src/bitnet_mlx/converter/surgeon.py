# src/bitnet_mlx/converter/surgeon.py
import json
import shutil
import glob
import logging
from pathlib import Path
import mlx.nn as nn
import mlx.core as mx
from huggingface_hub import snapshot_download
from ..core.dynamic_bitlinear import DynamicBitLinear

logger = logging.getLogger("TopologySurgeon")

class TopologySurgeon:
    @staticmethod
    def transmute(module: nn.Module) -> nn.Module:
        """Replaces FP16 linear manifolds with 1.58-bit DynamicBitLinear modules safely."""
        skip_substrs = [
            "lm_head", "embed", "gate", "vision_proj", "multi_modal_projector", 
            "conv", "patch_embedding", "wte", "wpe", "norm", "ln_", "audio_encoder", "mel"
        ]
        def _replace(mod: nn.Module, prefix: str = "") -> None:
            for name, child in list(mod.named_children()):
                path = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.Linear) and not any(k in path.lower() for k in skip_substrs):
                    setattr(mod, name, DynamicBitLinear(child.weight.shape[1], child.weight.shape[0], bias=child.bias is not None))
                else:
                    _replace(child, path)
        _replace(module)
        return module

    @staticmethod
    def build_manifold(repo_id: str, output_dir: str) -> None:
        try:
            from mlx_lm.utils import load
            model, _ = load(repo_id)
            quantized_model = TopologySurgeon.transmute(model)
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            
            mx.save_safetensors(str(out_path / "model.safetensors"), dict(quantized_model.parameters()))
            local_dir = snapshot_download(repo_id, allow_patterns=["*.json", "*.model", "tokenizer*", "preprocessor*"])
            for file in glob.glob(f"{local_dir}/*"):
                shutil.copy(file, out_path)
            
            with open(out_path / "config.json", "r+") as f:
                config = json.load(f)
                config["quantization"] = {"group_size": 0, "bits": 2, "type": "bitnet-1.58-omni-tda"}
                f.seek(0)
                json.dump(config, f, indent=2)
                f.truncate()
        except Exception:
            logger.exception("Manifold build failed.")
            raise