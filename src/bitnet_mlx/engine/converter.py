import mlx.nn as nn
import mlx.core as mx
from mlx_lm.utils import load, save_weights
from ..layers.bitnet import DynamicBitLinear
from pathlib import Path
import json
import shutil
import glob

class TopologySurgeon:
    @staticmethod
    def transmute(module: nn.Module) -> nn.Module:
        """Surgical routing: Replaces FP16 Linear manifolds with 1.58-bit DynamicBitLinear."""
        def _replace(mod, prefix=""):
            for name, child in list(mod.named_children()):
                path = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.Linear) and not any(k in path.lower() for k in ["lm_head", "embed", "gate"]):
                    in_d = child.weight.shape[1]
                    out_d = child.weight.shape[0]
                    has_bias = child.bias is not None
                    setattr(mod, name, DynamicBitLinear(in_d, out_d, bias=has_bias))
                else:
                    _replace(child, path)
        _replace(module)
        return module

    @staticmethod
    def build_bitnet_manifold(repo_id: str, output_dir: str):
        print(f"[*] Ingesting base FP16 manifold from: {repo_id}")
        model, tokenizer = load(repo_id)
        
        print("[*] Executing Topological Substitution (Zero-SVD)...")
        quantized_model = TopologySurgeon.transmute(model)
        
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        print(f"[*] Synchronizing safetensors to {output_dir}")
        save_weights(str(out_path), quantized_model.parameters())
        
        # Mirror original configuration mapping
        try:
            from huggingface_hub import snapshot_download
            local_dir = snapshot_download(repo_id, allow_patterns=["*.json", "*.model", "tokenizer*"])
            for file in glob.glob(f"{local_dir}/*"):
                shutil.copy(file, out_path)
            
            with open(out_path / "config.json", "r+") as f:
                config = json.load(f)
                config["quantization"] = {"group_size": 0, "bits": 2}
                f.seek(0)
                json.dump(config, f, indent=2)
                f.truncate()
        except Exception as e:
            print(f"[-] Config override bypassed. Proceeding with raw weights. {e}")
            
        print("[+] Sovereign quantization sequence complete.")
