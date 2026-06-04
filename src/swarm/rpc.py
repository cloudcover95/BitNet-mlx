# path: src/swarm/rpc.py
import zmq
import json
import logging
import mlx.core as mx
from typing import Optional

logger = logging.getLogger("JuniorSwarm")

class SwarmNode:
    """ZeroMQ Pipeline Parallelism Node for distributing DynamicBitLinear executions."""

    def __init__(self, port: int = 5555, is_worker: bool = True):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP if is_worker else zmq.REQ)
        self.port = port
        self.is_worker = is_worker

        if self.is_worker:
            self.socket.bind(f"tcp://*:{self.port}")
            logger.info(f"Swarm Worker mounted on port {self.port}. Awaiting sparse activation tensors...")
        else:
            logger.info(f"Swarm Dispatcher initialized. Target port: {self.port}")

    def connect(self, host: str):
        if not self.is_worker:
            self.socket.connect(f"tcp://{host}:{self.port}")
            logger.info(f"Connected to Swarm Mesh at {host}:{self.port}")

    def dispatch_tensor(self, x: mx.array) -> mx.array:
        """Serializes and dispatches an MLX array over the mesh."""
        shape = x.shape
        data = x.tolist()
        payload = json.dumps({"shape": shape, "data": data}).encode('utf-8')
        self.socket.send(payload)
        response = self.socket.recv()
        res_dict = json.loads(response.decode('utf-8'))
        return mx.array(res_dict["data"]).reshape(res_dict["shape"])

    def listen(self, execute_fn):
        """Worker loop listening for incoming sparse tensors."""
        while True:
            message = self.socket.recv()
            req_dict = json.loads(message.decode('utf-8'))
            x = mx.array(req_dict["data"]).reshape(req_dict["shape"])
            out = execute_fn(x)
            res_payload = json.dumps({"shape": out.shape, "data": out.tolist()}).encode('utf-8')
            self.socket.send(res_payload)
