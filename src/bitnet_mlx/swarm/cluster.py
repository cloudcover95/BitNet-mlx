# src/bitnet_mlx/swarm/cluster.py
import zmq
import zmq.asyncio
import asyncio
import logging
from typing import Dict, Any
from rich.console import Console
from rich.live import Live
from rich.table import Table

console = Console()
logger = logging.getLogger("JuniorSwarm")

class SwarmOrchestrator:
    """ZeroMQ Pipeline Parallelism architecture for infinite scaling."""
    def __init__(self, port: int = 5555):
        self.context = zmq.asyncio.Context()
        self.port = port
        self.workers: Dict[str, Dict[str, Any]] = {}

    def _generate_dashboard(self) -> Table:
        table = Table(title="JuniorSwarm: Active Node Topology")
        table.add_column("Node ID", style="cyan")
        table.add_column("RAM (GB)", style="green")
        table.add_column("Status", style="magenta")
        table.add_column("Assigned Shards", style="yellow")
        for node_id, data in self.workers.items():
            table.add_row(node_id, f"{data['ram_gb']:.1f}", data['status'], str(data['shards']))
        return table

    async def start(self) -> None:
        socket = self.context.socket(zmq.REP)
        socket.bind(f"tcp://*:{self.port}")
        console.print(f"[bold magenta][*] Swarm Orchestrator bound to tcp://*:{self.port}[/bold magenta]")
        
        with Live(self._generate_dashboard(), refresh_per_second=2) as live:
            try:
                while True:
                    msg = await socket.recv_json()
                    action = msg.get("action")
                    
                    if action == "register":
                        n_id = msg.get("node_id")
                        self.workers[n_id] = {"ram_gb": msg.get("ram_gb", 0), "status": "Ready", "shards": len(self.workers)+1}
                        await socket.send_json({"status": "ACK", "assigned_shard": self.workers[n_id]["shards"]})
                        live.update(self._generate_dashboard())
                    else:
                        await socket.send_json({"status": "ERR", "msg": "Unknown action"})
            except asyncio.CancelledError:
                logger.info("Orchestrator halting.")
            finally:
                socket.close()

class SwarmWorker:
    def __init__(self, target_ip: str, port: int = 5555):
        self.context = zmq.asyncio.Context()
        self.target = f"tcp://{target_ip}:{port}"
        self.node_id = f"worker_{id(self)}"

    async def register(self, ram_gb: float) -> None:
        socket = self.context.socket(zmq.REQ)
        socket.connect(self.target)
        try:
            await socket.send_json({"action": "register", "node_id": self.node_id, "ram_gb": ram_gb})
            reply = await socket.recv_json()
            console.print(f"[bold green][+] Swarm ACK: Operating on Shard {reply.get('assigned_shard')}.[/bold green]")
        finally:
            socket.close()