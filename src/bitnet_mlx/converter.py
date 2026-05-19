import mlx.core as mx
import os, time, logging, shutil
from pathlib import Path
from huggingface_hub import snapshot_download
from .bitnet_layers import compute_channel_ternary

logger = logging.getLogger("BitNet.Converter")
logging.basicConfig(level=logging.INFO, format='%(message)s')

class BitNetCompiler:
    def __init__(self):
        self.exclude_layers = {"embed", "lm_head", "gate"}

    def _clone_assets(self, src: Path, dest: Path):
        for ext in ["*.json", "*.model", "*.txt"]:
            for file in src.glob(ext):
                if "safetensors.index" in file.name: continue 
                shutil.copy(file, dest / file.name)

    def convert_to_ternary(self, model_id_or_path: str, output_dir: str):
        if os.path.exists(model_id_or_path):
            src_path = Path(model_id_or_path)
        else:
            logger.info(f"[*] Fetching HF Manifold: {model_id_or_path}")
            src_path = Path(snapshot_download(repo_id=model_id_or_path, local_dir_use_symlinks=False))

        dest_path = Path(output_dir)
        os.makedirs(dest_path, exist_ok=True)
        
        logger.info(f"[*] Igniting AbsMean PTQ Engine on: {src_path}")
        t0 = time.perf_counter()
        
        self._clone_assets(src_path, dest_path)
        st_files = list(src_path.glob("*.safetensors"))

        if not st_files:
            logger.error("[-] Fatal: No Safetensors located.")
            return

        for shard in st_files:
            logger.info(f"[*] Compressing Shard -> Ternary Space: {shard.name}")
            weights = mx.load(str(shard))
            ternary_dict = {}
            for name, tensor in weights.items():
                if "weight" in name and len(tensor.shape) == 2 and not any(k in name.lower() for k in self.exclude_layers):
                    base_name = name.rsplit('.weight', 1)[0]
                    w_q, gamma = compute_channel_ternary(tensor)
                    mx.eval(w_q, gamma)
                    ternary_dict[f"{base_name}.w_q"], ternary_dict[f"{base_name}.gamma"] = w_q, gamma
                else:
                    ternary_dict[name] = tensor
                    
            if hasattr(mx, 'clear_cache'): mx.clear_cache()
            mx.save_safetensors(str(dest_path / shard.name), ternary_dict)

        logger.info(f"[+] Sub-2-Bit Conversion Complete ({time.perf_counter()-t0:.2f}s). Manifold saved to {output_dir}")
