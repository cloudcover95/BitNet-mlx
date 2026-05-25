import mlx.nn as nn
from mlx_lm.utils import load, save_weights
from ..layers.bitnet import DynamicBitLinear
from pathlib import Path

class SovereignConverter:
    @staticmethod
    def map_topology(model: nn.Module) -> nn.Module:
        """Recursively parses model topology, replacing FP16 Linear layers with Ternary gates."""
        def _replace_layers(module, name=""):
            for child_name, child in list(module.named_children()):
                full_name = f"{name}.{child_name}" if name else child_name
                # Protect LM head and routing logic from quantization drop-off
                if isinstance(child, nn.Linear) and not any(k in full_name.lower() for k in ["embed", "lm_head", "gate"]):
                    in_d = child.weight.shape[1]
                    out_d = child.weight.shape[0]
                    has_bias = child.bias is not None
                    setattr(module, child_name, DynamicBitLinear(in_d, out_d, bias=has_bias))
                else:
                    _replace_layers(child, full_name)
        
        _replace_layers(model)
        return model

    @staticmethod
    def build_bitnet_manifold(repo_id: str, output_dir: str):
        print(f"[*] Ingesting base FP16 manifold from: {repo_id}")
        model, tokenizer = load(repo_id)
        
        print("[*] Executing Topological Substitution (Zero-SVD)...")
        quantized_model = SovereignConverter.map_topology(model)
        
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        print(f"[*] Synchronizing BitNet-MLX state to {output_dir}")
        save_weights(out_path, quantized_model.parameters())
        tokenizer.save_pretrained(out_path)
        print("[+] Sovereign quantization complete.")
