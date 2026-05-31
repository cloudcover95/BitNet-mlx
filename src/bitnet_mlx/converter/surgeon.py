# src/bitnet_mlx/converter/surgeon.py
import json
import shutil
import glob
import logging
from pathlib import Path
import mlx.nn as nn
import mlx.core as mx
from mlx_lm.utils import load
from huggingface_hub import snapshot_download
from ..core.dynamic_bitlinear import DynamicBitLinear
from rich.console import Console

console = Console()
logger = logging.getLogger("TopologySurgeon")

class TopologySurgeon:
    @staticmethod
    def transmute(module: nn.Module) -> nn.Module:
        """
        Replaces FP16 manifolds with 1.58-bit ternary limits.
        VLM-Aware: Isolates multimodal projectors, convolutions, and embeddings.
        """
        skip_substrs = [
            "lm_head", "embed", "gate", "vision_proj", "multi_modal_projector", 
            "conv", "patch_embedding", "wte", "wpe", "norm", "ln_"
        ]
        
        def _replace(mod: nn.Module, prefix: str = "") -> None:
            for name, child in list(mod.named_children()):
                path = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.Linear) and not any(k in path.lower() for k in skip_substrs):
                    in_d, out_d = child.weight.shape[1], child.weight.shape[0]
                    setattr(mod, name, DynamicBitLinear(in_d, out_d, bias=child.bias is not None))
                else:
                    _replace(child, path)
        _replace(module)
        return module

    @staticmethod
    def build_manifold(repo_id: str, output_dir: str) -> None:
        try:
            console.print(f"[*] Ingesting FP16 topology from [bold cyan]{repo_id}[/bold cyan]...")
            model, _ = load(repo_id)
            
            console.print("[*] Executing VLM-Safe Topological Transmutation...")
            quantized_model = TopologySurgeon.transmute(model)
            
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            
            console.print("[*] Serializing MLX Safetensors...")
            mx.save_safetensors(str(out_path / "model.safetensors"), dict(quantized_model.parameters()))
            
            # Fetch config/tokenizer assets bypassing heavy tensor files
            local_dir = snapshot_download(repo_id, allow_patterns=["*.json", "*.model", "tokenizer*"])
            for file in glob.glob(f"{local_dir}/*"):
                shutil.copy(file, out_path)
            
            with open(out_path / "config.json", "r+") as f:
                config = json.load(f)
                config["quantization"] = {"group_size": 0, "bits": 2, "type": "bitnet-1.58-omni-tda"}
                f.seek(0)
                json.dump(config, f, indent=2)
                f.truncate()
                
            console.print(f"[bold green][+] Transmutation successful. Matrix secured at {output_dir}[/bold green]")
        except Exception as e:
            console.print(f"[bold red][-] Surgical failure: {e}[/bold red]")
            logger.exception("Manifold build failed.")
            raise