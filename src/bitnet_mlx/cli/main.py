# src/bitnet_mlx/cli/main.py
import typer
from rich.console import Console
from ..sandbox.suite import SandboxSuite
from ..converter.surgeon import TopologySurgeon

app = typer.Typer(help="BitNet-MLX Omni-Sovereign SDK by JuniorCloud LLC")
console = Console()

@app.command()
def sandbox():
    """Verify SDK TDA Variance integrity, and 5-Discipline Training engine."""
    if not SandboxSuite.execute_validation_matrix(): raise typer.Exit(code=1)

@app.command()
def convert(repo: str, output: str):
    """Transmute HF repository to BitNet-MLX 1.58-bit."""
    TopologySurgeon.build_manifold(repo, output)

if __name__ == "__main__":
    app()