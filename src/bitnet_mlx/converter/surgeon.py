# src/bitnet_mlx/converter/surgeon.py
import json
import shutil
import glob
import logging
import math
from pathlib import Path
import mlx.nn as nn
import mlx.core as mx
from mlx_lm.utils import load
from huggingface_hub import snapshot_download
from ..core.dynamic_bitlinear import DynamicBitLinear
from ..utils.hardware import settings
from rich.console import Console

console = Console()
logger = logging.getLogger("TopologySurgeon")

class TopologySurgeon:
    @staticmethod
    def transmute(module: nn.Module) -> nn.Module:
        """VLM & Audio Aware Topology Surgeon."""
        skip_substrs = [
            "lm_head", "embed", "gate", "vision_proj", "multi_modal_projector", 
            "conv", "patch_embedding", "wte", "wpe", "norm", "ln_",
            "audio_encoder", "mel"
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
    def export_sharded_safetensors(model: nn.Module, output_dir: Path) -> None:
        """Serializes tensors into chunks to prevent memory saturation on Swarm Worker ingestion."""
        weights = dict(model.parameters())
        max_bytes = settings.max_shard_size_gb * (1024 ** 3)
        
        current_chunk = {}
        current_size = 0
        chunk_idx = 1
        index_map = {"metadata": {"total_size": 0}, "weight_map": {}}
        
        for k, v in weights.items():
            tensor_size = v.size * v.itemsize
            if current_size + tensor_size > max_bytes and current_chunk:
                fname = f"model-{chunk_idx:05d}.safetensors"
                mx.save_safetensors(str(output_dir / fname), current_chunk)
                current_chunk = {}
                current_size = 0
                chunk_idx += 1
                
            current_chunk[k] = v
            current_size += tensor_size
            index_map["weight_map"][k] = f"model-{chunk_idx:05d}.safetensors"
            index_map["metadata"]["total_size"] += tensor_size
            
        if current_chunk:
            fname = f"model-{chunk_idx:05d}.safetensors"
            mx.save_safetensors(str(output_dir / fname), current_chunk)
            
        with open(output_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index_map, f, indent=2)

    @staticmethod
    def build_manifold(repo_id: str, output_dir: str) -> None:
        try:
            console.print(f"[*] Ingesting FP16 topology from [bold cyan]{repo_id}[/bold cyan]...")
            model, _ = load(repo_id)
            
            console.print("[*] Executing Topological Transmutation...")
            quantized_model = TopologySurgeon.transmute(model)
            
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            
            console.print("[*] Sharding MLX Safetensors for JuniorSwarm Ingestion...")
            TopologySurgeon.export_sharded_safetensors(quantized_model, out_path)
            
            local_dir = snapshot_download(repo_id, allow_patterns=["*.json", "*.model", "tokenizer*", "preprocessor*"])
            for file in glob.glob(f"{local_dir}/*"):
                shutil.copy(file, out_path)
            
            with open(out_path / "config.json", "r+") as f:
                config = json.load(f)
                config["quantization"] = {"group_size": 0, "bits": 2, "type": "bitnet-1.58-omni-tda"}
                f.seek(0)
                json.dump(config, f, indent=2)
                f.truncate()
                
            console.print(f"[bold green][+] Transmutation secured at {output_dir}[/bold green]")
        except Exception as e:
            logger.exception("Manifold build failed.")
            raise