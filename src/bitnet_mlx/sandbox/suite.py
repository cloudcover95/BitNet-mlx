# src/bitnet_mlx/sandbox/suite.py
import mlx.core as mx
import mlx.nn as nn
from rich.console import Console
from rich.table import Table
from ..core.dynamic_bitlinear import DynamicBitLinear
from ..core.quantization import compute_asymmetric_ternary_ste
from ..core.blackbox_ops import BlackBoxArchitecture
from ..utils.hardware import HardwareGovernor
from ..training.engine import SovereignTrainer

console = Console()

def mock_dataset_generator(batches=1, bsz=1, seq=128, dim=64):
    for _ in range(batches):
        yield mx.random.normal((bsz, seq), dtype=mx.float16), mx.random.normal((bsz, dim), dtype=mx.float16)

class SandboxSuite:
    @staticmethod
    def execute_validation_matrix() -> bool:
        console.print("[bold yellow][*] Booting Sovereign Training & Inference Sandbox...[/bold yellow]")
        
        hw_model, hw_brand = HardwareGovernor.get_soc_topology()
        q_max = HardwareGovernor.calculate_quantization_ceiling()
        console.print(f"\n[bold cyan]--- Edge Telemetry ---[/bold cyan]\nHardware: {hw_model} ({hw_brand}) | Active q_max: {q_max}")
        
        console.print("\n[bold cyan]--- Inference Emulation: TDA Math Check ---[/bold cyan]")
        table = Table("Manifold", "Fidelity", "Laplacian Div", "Sparsity")
        for dim in [512, 1024]:
            p = mx.random.normal((dim, dim), dtype=mx.float16) * 0.05
            w_q_ste, _, _, w_raw = compute_asymmetric_ternary_ste(p)
            var_ret = min(mx.var(w_q_ste.astype(mx.float32)).item() / (mx.var(p.astype(mx.float32)).item() + 1e-9), 1.0)
            tda_div = BlackBoxArchitecture.laplacian_spectral_tda(p.astype(mx.float32), w_q_ste.astype(mx.float32)).item()
            sparsity = mx.sum(w_raw == 0).item() / (dim * dim)
            table.add_row(f"{dim}x{dim}", f"{var_ret*100:.2f}%", f"{tda_div:.4f}", f"{sparsity*100:.2f}%")
            if var_ret < 0.90: 
                console.print(f"[bold red][!] Variance Fidelity failure on {dim}x{dim}[/bold red]")
                return False
        console.print(table)
        
        console.print("\n[bold cyan]--- Training Emulation: 5-Discipline Validation ---[/bold cyan]")
        for mode in ["full", "qat", "lora", "qlora", "ptq_recovery"]:
            mock_model = nn.Sequential(DynamicBitLinear(128, 64), DynamicBitLinear(64, 64))
            try:
                SovereignTrainer.execute_cycle(mock_model, dataset=mock_dataset_generator(), mode=mode, epochs=1)
            except Exception as e:
                console.print(f"[bold red]Root Node Failure in {mode}: {str(e)}[/bold red]")
                return False
        
        return True