import zmq
import asyncio
import logging
from rich.console import Console

console = Console()
logger = logging.getLogger("JuniorSwarm")

class JuniorSwarmNode:
    """
    Decentralized ZeroMQ Orchestrator for JuniorCloud Subnets.
    Allows A-series and M-series nodes to interconnect over 48V networks.
    """
    def __init__(self, port: int = 5555, is_worker: bool = False):
        self.context = zmq.asyncio.Context()
        self.port = port
        self.is_worker = is_worker

    async def start_orchestrator(self) -> None:
        socket = self.context.socket(zmq.REP)
        socket.bind(f"tcp://*:{self.port}")
        console.print(f"[bold magenta][*] JuniorSwarm Orchestrator bound to port {self.port}...[/bold magenta]")
        
        while True:
            msg = await socket.recv_json()
            console.print(f"[dim]Swarm REQ received: {msg['task_id']}[/dim]")
            await asyncio.sleep(0.1) # Simulate tensor routing
            await socket.send_json({"status": "ACK", "payload": "Tensor block queued."})

    async def start_worker(self, target_ip: str) -> None:
        socket = self.context.socket(zmq.REQ)
        socket.connect(f"tcp://{target_ip}:{self.port}")
        console.print(f"[bold cyan][*] JuniorSwarm Worker connected to {target_ip}:{self.port}...[/bold cyan]")
        
        await socket.send_json({"task_id": "ping", "capability": "A17_Pro"})
        reply = await socket.recv_json()
        console.print(f"[bold green][+] Swarm ACK: {reply}[/bold green]")
