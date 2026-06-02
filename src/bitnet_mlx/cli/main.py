import typer
import uvicorn
import asyncio
from rich.console import Console
from ..converter.surgeon import TopologySurgeon
from ..inference.engine import SovereignInference
from ..inference.mcp_server import ObsidianVaultMCP

app = typer.Typer(help="BitNet-MLX v87.0.0 Sovereign SDK by JuniorCloud LLC")
console = Console()

@app.command()
def info():
    """Display SDK status and architecture highlights."""
    console.print("[bold green]BitNet-MLX v87.0.0 — Sovereign Engine Active[/bold green]")
    console.print("• Python >= 3.9 Runtime System: Engaged")
    console.print("• 2-Bit Streaming KV-Cache Manifold: Persisted")
    console.print("• Obsidian Vault MCP Protocol: Mounted")

@app.command()
def convert(repo: str, output: str):
    """Transmute HF repository to BitNet-MLX 1.58-bit."""
    TopologySurgeon.build_manifold(repo, output)

@app.command()
def chat(model: str, prompt: str, max_tokens: int = 512):
    """Execute streaming inference directly from terminal."""
    for chunk in SovereignInference.stream_inference(model, prompt, max_tokens):
        print(chunk, end="", flush=True)
    print()

@app.command()
def serve(host: str = "127.0.0.1", port: int = 8080):
    """Mount the production FastAPI SSE OpenAI-compatible endpoint."""
    console.print(f"[bold yellow]Mounting Sovereign API at http://{host}:{port}/v1/chat/completions[/bold yellow]")
    uvicorn.run("bitnet_mlx.inference.server:SovereignAPI", host=host, port=port, log_level="info")

@app.command()
def mcp(vault_path: str = "./JuniorCoach-Vault"):
    """Mounts the Model Context Protocol (MCP) server over STDIO for Obsidian integration."""
    mcp_server = ObsidianVaultMCP(vault_path)
    asyncio.run(mcp_server.run_stdio())

if __name__ == "__main__":
    app()
