# src/bitnet_mlx/cli/main.py
import typer
import asyncio
import uvicorn
import bitnet_mlx.cli.api as api_module
from rich.console import Console
from ..converter.surgeon import TopologySurgeon
from ..inference.engine import InjectionEngine
from ..sandbox.suite import JuniorSandboxSuite

app = typer.Typer(help="BitNet-MLX Omni-Sovereign SDK by JuniorCloud LLC")
console = Console()

@app.command()
def convert(repo: str, output: str):
    """HF -> MLX 1.58-bit Topological Transmutation."""
    TopologySurgeon.build_manifold(repo, output)

@app.command()
def chat(model: str, prompt: str, max_tokens: int = 512):
    """Autonomous Sovereign Async Inference."""
    console.print(f"[bold green][*] Booting Sovereign Logic Engine...[/bold green]\n")
    async def _stream():
        async for chunk in InjectionEngine.execute_stream(model, prompt, max_tokens):
            print(chunk, end="", flush=True)
    asyncio.run(_stream())

@app.command()
def sandbox():
    """Boot the JuniorCloud ecosystem validation matrix."""
    if not JuniorSandboxSuite.execute_validation_matrix():
        raise typer.Exit(code=1)

@app.command()
def serve(model: str, host: str = "127.0.0.1", port: int = 8080):
    """Mount the asynchronous InjectionEngine to a FastAPI OpenAI-compatible node."""
    console.print(f"[bold green][*] Mounting Sovereign API on {host}:{port} with manifold: {model}[/bold green]")
    api_module.ACTIVE_MODEL_PATH = model
    uvicorn.run(api_module.api, host=host, port=port, log_level="info")

if __name__ == "__main__":
    app()