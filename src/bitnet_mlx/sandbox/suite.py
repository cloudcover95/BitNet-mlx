# src/bitnet_mlx/sandbox/suite.py
import mlx.core as mx
import pandas as pd
from rich.console import Console
from rich.table import Table
from ..core.quantization import compute_tda_ternary_ste
from ..core.tda_math import TopologicalManifold
from ..utils.hardware import HardwareGovernor

console = Console()

class JuniorSandboxSuite:
    """Fuses JuniorStock predictive manifolds, JuniorQuant TDA metrics, and JuniorHome telemetry."""
    @staticmethod
    def execute_validation_matrix() -> bool:
        console.print("[bold yellow][*] Booting JuniorCloud Omni-Sovereign Sandbox...[/bold yellow]")
        
        soc = HardwareGovernor.get_soc_topology()
        cpu, mem, swap = HardwareGovernor.get_system_load()
        q_ceiling = HardwareGovernor.calculate_quantization_ceiling()
        
        console.print(f"\n[bold cyan]--- JuniorHome Edge Telemetry ---[/bold cyan]")
        console.print(f"Topology: [white]{soc}[/white] | Active TDA Ceiling (q_max): [bold green]{q_ceiling}[/bold green]")
        console.print(f"Compute Load: {cpu}% | RAM Free: {mem:.2f}GB | Swap: {swap:.2f}GB")
        
        console.print("\n[bold cyan]--- JuniorQuant Spectral Variance Inference ---[/bold cyan]")
        table = Table(show_header=True)
        table.add_column("Manifold", justify="right", style="cyan")
        table.add_column("Fidelity", justify="right", style="green")
        table.add_column("Spectral Gap Div", justify="right", style="magenta")
        
        results = []
        for dim in [512, 1024, 2048, 4096]:
            p = mx.random.normal((dim, dim), dtype=mx.float16) * 0.05
            w_q_ste, _, _ = compute_tda_ternary_ste(p)
            
            var_ret = min(mx.var(w_q_ste.astype(mx.float32)).item() / (mx.var(p.astype(mx.float32)).item() + 1e-9), 1.0)
            tda_div = TopologicalManifold.compute_manifold_divergence(p.astype(mx.float32), w_q_ste.astype(mx.float32)).item()
            
            results.append({"dim": dim, "fidelity": var_ret, "tda_div": tda_div})
            table.add_row(f"{dim}x{dim}", f"{var_ret*100:.2f}%", f"{tda_div:.4f}")
            
            if var_ret < 0.99:
                console.print(f"[bold red][!] Fatal Variance Collapse on {dim}x{dim}.[/bold red]")
                return False
                
        console.print(table)
        console.print("\n[bold green][+] JuniorCoach backend matrix is fully stable and sovereign.[/bold green]")
        return True