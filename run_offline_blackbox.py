# path: BitNet-mlx/run_offline_blackbox.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from src.inference.proprietary_blackbox import SovereignBitNetInference

app = FastAPI(title="JCLLC BitNet-MLX Proprietary Blackbox")
engine = SovereignBitNetInference()


class TensorPayload(BaseModel):
    ticker: str
    spatial_tensor: List[List[float]]  # Expected shape: (1, T)


@app.post("/v1/proprietary/infer")
async def execute_inference(payload: TensorPayload):
    """
    Ingests the raw 2D spatial tensor.
    Returns the computed KPZ/SVD manifold state and 1.58-bit text consensus.
    """
    result = engine.generate_consensus(payload.ticker, payload.spatial_tensor)
    return result


if __name__ == "__main__":
    # Binds strictly to localhost to prevent external subnet access
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")
