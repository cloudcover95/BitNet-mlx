# src/bitnet_mlx/cli/main.py
import typer
from rich.console import Console
from ..sandbox.suite import SandboxSuite
from ..training.engine import TrainingEngine

app = typer.Typer(help="BitNet-MLX Omni-Sovereign SDK by JuniorCloud LLC")
console = Console()

@app.command()
def sandbox():
    """Verify SDK ZK Cryptography, DMAA, TDA Variance integrity, and Training engine."""
    if not SandboxSuite.execute_validation_matrix(): raise typer.Exit(code=1)

@app.command()
def train(mode: str = typer.Option("qlora", help="full, qat, lora, qlora, ptq_recovery"), epochs: int = 5, lr: float = 1e-4):
    """Execute specialized Sovereign Training loops natively."""
    import mlx.nn as nn
    from ..core.dynamic_bitlinear import DynamicBitLinear
    try:
        # Emulated mock model for CLI trigger demonstration
        model = nn.Sequential(DynamicBitLinear(256, 128), DynamicBitLinear(128, 64))
        TrainingEngine.run_training_cycle(model, mode=mode, epochs=epochs, lr=lr)
    except Exception as e:
        console.print(f"[bold red]Training fault: {e}[/bold red]")

if __name__ == "__main__":
    app()