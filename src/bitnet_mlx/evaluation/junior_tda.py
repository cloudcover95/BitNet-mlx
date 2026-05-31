import mlx.core as mx
import pandas as pd
from pathlib import Path
from ..core.quantization import compute_tda_ternary_ste
from ..core.tda_math import TopologicalManifold
from rich.console import Console
from rich.table import Table

console = Console()

class JuniorCloudTDAEvaluator:
    @staticmethod
    def run_validation_suite() -> bool:
        Path("logs").mkdir(exist_ok=True)
        results = []
        
        table = Table(title="Omni-Sovereign TDA: Spectral Variance Inference", show_header=True)
        table.add_column("Manifold", justify="right", style="cyan")
        table.add_column("Fidelity", justify="right", style="green")
        table.add_column("TDA Div", justify="right", style="magenta")
        table.add_column("Sparsity", justify="right", style="yellow")
        
        for dim in [512, 1024, 2048]:
            p = mx.random.normal((dim, dim), dtype=mx.float16) * 0.05
            w_q_ste, _, w_raw = compute_tda_ternary_ste(p)
            
            var_ret = min(mx.var(w_q_ste.astype(mx.float32)).item() / (mx.var(p.astype(mx.float32)).item() + 1e-9), 1.0)
            tda_div = TopologicalManifold.compute_manifold_divergence(p.astype(mx.float32), w_q_ste.astype(mx.float32)).item()
            sparsity = mx.sum(w_raw == 0).item() / (dim * dim)
            
            results.append({
                "manifold_dim": dim, "variance_fidelity": var_ret, 
                "spectral_divergence": tda_div, "sparsity_ratio": sparsity
            })
            table.add_row(f"{dim}x{dim}", f"{var_ret*100:.2f}%", f"{tda_div:.4f}", f"{sparsity*100:.2f}%")
            
            if var_ret < 0.99:
                console.print(f"[bold red][!] Fatal Variance Collapse on {dim}x{dim}.[/bold red]")
                return False
                
        console.print(table)
        pd.DataFrame(results).to_parquet("logs/omni_emulation_drift.parquet", index=False)
        return True
