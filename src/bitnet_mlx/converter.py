import mlx.core as mx
import os, logging, shutil
from pathlib import Path
from huggingface_hub import snapshot_download
from .bitnet_layers import compute_hybrid_ternary_ste
from .telemetry import StructuralEvaluator, HighDensityLogger
from .autotune import GrokAutoTuner

logger = logging.getLogger("BitNet.Compiler")

class BitNetCompiler:
def init(self, enable_autotune: bool = True):
self.exclude_layers = {"embed", "lm_head", "gate"}
self.enable_autotune = enable_autotune
self.hd_logger = HighDensityLogger()

def compile_manifold(self, model_id_or_path: str, output_dir: str):
    src_path = Path(model_id_or_path) if os.path.exists(model_id_or_path) else Path(snapshot_download(repo_id=model_id_or_path, local_dir_use_symlinks=False))
    dest_path = Path(output_dir)
    os.makedirs(dest_path, exist_ok=True)
    
    for file in src_path.glob("*"):
        if file.suffix in [".json", ".model", ".txt"] and "safetensors.index" not in file.name:
            shutil.copy(file, dest_path / file.name)

    for shard in list(src_path.glob("*.safetensors")):
        logger.info(f"[*] Ingesting continuous dense tensor sets: {shard.name}")
        weights = mx.load(str(shard))
        ternary_dict = {}
        
        for name, tensor in weights.items():
            if "weight" in name and len(tensor.shape) == 2 and not any(k in name.lower() for k in self.exclude_layers):
                base_name = name.rsplit('.weight', 1)[0]
                adaptive_flag = False
                
                if self.enable_autotune:
                    optimized_w, w_q, gamma, w_outliers, used_limit = GrokAutoTuner.calculate_optimal_sparsity(tensor)
                    adaptive_flag = True
                else:
                    w_q, gamma, w_outliers = compute_hybrid_ternary_ste(tensor)
                    used_limit = 0.0
                    
                mx.eval(w_q, gamma, w_outliers)
                w_recon = (w_q * gamma) + w_outliers
                fid = min(mx.var(w_recon).item() / (mx.var(tensor).item() + 1e-9), 1.0)
                
                # C2V Byte Math: (w_q * 1 byte) + (gamma * 2 bytes) + (w_outliers * 2 bytes)
                compressed_size = (w_q.size * 1) + (gamma.size * 2) + (mx.sum(w_outliers != 0).item() * 2)
                ratio = float(tensor.size * 2) / float(max(compressed_size, 1))
                
                weightings = StructuralEvaluator.compute_parameter_weighting(w_q, w_outliers)
                self.hd_logger.append_layer_telemetry(name, tensor.shape, fid, ratio, weightings, used_limit if adaptive_flag else 0.0)
                
                ternary_dict[f"{base_name}.w_q"] = w_q
                ternary_dict[f"{base_name}.gamma"] = gamma
                ternary_dict[f"{base_name}.w_outliers"] = w_outliers
            else:
                ternary_dict[name] = tensor
                
        mx.save_safetensors(str(dest_path / shard.name), ternary_dict)

    self.hd_logger.commit_to_disk()
    logger.info("[+] Target sequence compiled. Telemetry published.")
