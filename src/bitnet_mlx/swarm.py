import mlx.core as mx
import numpy as np
import io
import logging
import httpx
from fastapi import FastAPI, Request
from typing import List

logger = logging.getLogger("BitNet.Swarm")
app = FastAPI(title="JuniorCloud Swarm Node")

class BinaryRPC:
@staticmethod
def pack_tensor(t: mx.array) -> bytes:
arr = np.array(t)
bio = io.BytesIO()
np.save(bio, arr)
return bio.getvalue()

@staticmethod
def unpack_tensor(data: bytes) -> mx.array:
    bio = io.BytesIO(data)
    return mx.array(np.load(bio))
class TensorRPC:
def init(self, peers: List[str]):
self.peers = peers
self.client = httpx.AsyncClient(timeout=10.0)

async def broadcast_gradient_sync(self, tensor: mx.array):
    payload = BinaryRPC.pack_tensor(tensor)
    for peer in self.peers:
        try:
            await self.client.post(
                f"http://{peer}/sync_gradients",
                content=payload,
                headers={"Content-Type": "application/octet-stream"}
            )
        except Exception as e:
            logger.error(f"[-] Peer {peer} detached from Swarm. {str(e)}")
@app.post("/sync_gradients")
async def sync_gradients(request: Request):
data = await request.body()
tensor = BinaryRPC.unpack_tensor(data)
logger.info(f"[+] Swarm Sync: Ingesting binary gradient fragment [{tensor.shape}].")
return {"status": "ACK", "matrix": "fused"}
