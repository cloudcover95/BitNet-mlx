from mcp.server.fastmcp import FastMCP
from ..core.evaluation import GrokEvaluator
import psutil

mcp = FastMCP("BitNet-MLX-Sovereign-Matrix")

@mcp.tool()
def bitnet_system_capability() -> str:
    cpu = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory()
    return f"UMA RAM: {ram.available / 1e9:.2f}GB. Load: {cpu}%"

def start_server():
    mcp.run()
