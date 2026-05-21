import mlx.core as mx
import numpy as np
import io
import logging
from fastapi import FastAPI, Request
import uvicorn

logger = logging.getLogger("BitNet.Swarm")
app = FastAPI(title="JuniorCloud Omni-Swarm Node")

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

@app.post("/sync_gradients")
async def sync_gradients(request: Request):
    data = await request.body()
    try:
        tensor = BinaryRPC.unpack_tensor(data)
        logger.info(f"[+] Swarm Sync: Ingesting binary gradient payload {tensor.shape}.")
        return {"status": "ACK", "matrix": "fused"}
    except Exception as e:
        logger.error(f"[-] Synchronization failure: {str(e)}")
        return {"status": "FAIL"}

def boot_swarm_listener(port: int = 9000):
    logger.info(f"[*] Booting Omni-Swarm RPC on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
