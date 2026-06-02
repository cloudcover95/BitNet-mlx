import json
import shutil
import glob
import logging
from pathlib import Path
from typing import List, Set
import mlx.nn as nn
import mlx.core as mx
from huggingface_hub import snapshot_download
from ..core.dynamic_bitlinear import DynamicBitLinear

logger = logging.getLogger("TopologySurgeon")

class TopologySurgeon:
    """Multi-modal safe 1.58-bit transmuter with rigorous VLM/Audio bypass."""

    PROTECTED_SUBSTRS: Set[str] = {
        "lm_head", "embed", "gate", "vision_proj", "multi_modal_projector",
        "conv", "patch_embedding", "wte", "wpe", "norm", "ln_",
        "audio_encoder", "mel", "whisper", "speech", "vision_tower", "image_newline"
    }

    @staticmethod
    def _is_protected(path: str) -> bool:
        lower = path.lower()
        return any(k in lower for k in TopologySurgeon.PROTECTED_SUBSTRS)

    @staticmethod
    def transmute(module: nn.Module) -> nn.Module:
        """Replace FP16 Linear with DynamicBitLinear except protected multi-modal paths."""
        def _replace(mod: nn.Module, prefix: str = "") -> None:
            for name, child in list(mod.named_children()):
                path = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.Linear) and not TopologySurgeon._is_protected(path):
                    new_layer = DynamicBitLinear(
                        in_d=child.weight.shape[1],
                        out_d=child.weight.shape[0],
                        bias=child.bias is not None
                    )
                    # Preserve original weights for STE path
                    new_layer.weight = child.weight
                    if child.bias is not None:
                        new_layer.bias = child.bias
                    setattr(mod, name, new_layer)
                else:
                    _replace(child, path)
        _replace(module)
        logger.info("Multi-modal bypass applied. VLM/Audio encoders protected.")
        return module

    @staticmethod
    def build_manifold(repo_id: str, output_dir: str) -> None:
        """Download, transmute with bypass, save 1.58-bit manifold + tokenizer/config."""
        try:
            from mlx_lm.utils import load, save_weights
            print(f"[*] Ingesting {repo_id} with multi-modal bypass...")
            model, tokenizer = load(repo_id)
            quantized = TopologySurgeon.transmute(model)
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            save_weights(out, dict(quantized.parameters()))
            local = snapshot_download(repo_id, allow_patterns=["*.json", "tokenizer*", "preprocessor*", "*.model"])
            for f in glob.glob(f"{local}/*"):
                shutil.copy(f, out)
            with open(out / "config.json", "r+") as f:
                cfg = json.load(f)
                cfg["quantization"] = {"group_size": 0, "bits": 2, "type": "bitnet-1.58-omni-tda"}
                f.seek(0)
                json.dump(cfg, f, indent=2)
                f.truncate()
            print(f"[+] Secured multi-modal 1.58-bit manifold at {output_dir}")
        except Exception as e:
            logger.exception("Manifold build failed")
            raise
