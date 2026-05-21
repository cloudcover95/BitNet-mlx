import mlx.core as mx
import os, logging, shutil, json
from pathlib import Path
from huggingface_hub import snapshot_download
from .bitnet_layers import compute_hybrid_ternary_ste
from .telemetry import StructuralEvaluator, HighDensityLogger
from .autotune import GrokAutoTuner
from .audit import EmulationAuditor

logger = logging.getLogger("BitNet.Compiler")

class BitNetCompiler:
def init(self, enable_autotune: bool = True):
self.exclude_layers = {"embed", "lm_head", "gate"}
self.enable_autotune = enable_autotune
self.hd_logger = HighDensityLogger()

def compile_manifold(self, model_id_or_path: str, output_dir: str):
    if not EmulationAuditor().execute_audit():
        logger.error("[-] Hardware-bound audit failed. Compilation halted.")
        return

    src_path = Path(model_id_or_path) if os.path.exists(model_id_or_path) else Path(snapshot_download(repo_id=model_id_or_path, local_dir_use_symlinks=False))
    dest_path = Path(output_dir)
    os.makedirs(dest_path, exist_ok=True)
    
    for file in src_path.glob("*"):
        if file.suffix in [".json", ".model", ".txt"] and "safetensors.index" not in file.name:
            shutil.copy(file, dest_path / file.name)

    total_params, zeros, pos_ones, neg_ones, outliers = 0, 0, 0, 0, 0

    for shard in list(src_path.glob("*.safetensors")):
        logger.info(f"[*] Compiling Aegis-Omni logic gates: {shard.name}")
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
                    
                w_q_eval = mx.round(w_q).astype(mx.int8)
                total_params += tensor.size
                zeros += mx.sum(w_q_eval == 0).item()
                pos_ones += mx.sum(w_q_eval == 1).item()
                neg_ones += mx.sum(w_q_eval == -1).item()
                outliers += mx.sum(w_outliers != 0).item()
                
                fid = min(mx.var((w_q * gamma) + w_outliers).item() / (mx.var(tensor).item() + 1e-9), 1.0)
                compressed_size = (w_q.size * 1) + (gamma.size * 2) + (mx.sum(w_outliers != 0).item() * 2)
                ratio = float(tensor.size * 2) / float(max(compressed_size, 1))
                
                weightings = StructuralEvaluator.compute_parameter_weighting(w_q, w_outliers)
                self.hd_logger.append_layer_telemetry(name, tensor.shape, fid, ratio, weightings, used_limit if adaptive_flag else 0.0)
                
                ternary_dict[f"{base_name}.weight"] = optimized_w if self.enable_autotune else tensor
            else:
                ternary_dict[name] = tensor
        
        metadata = {
            "format": "juniorcloud-bitnet-v20-aegis-omni",
            "c2v_total_params": str(total_params),
            "c2v_zeros": str(zeros),
            "c2v_pos_ones": str(pos_ones),
            "c2v_neg_ones": str(neg_ones),
            "c2v_outliers": str(outliers)
        }
        mx.save_safetensors(str(dest_path / shard.name), ternary_dict, metadata=metadata)

    self.hd_logger.commit_to_disk()
    logger.info("[+] Target sequence compiled. Self-Reporting Metadata injected.")
