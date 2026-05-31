# src/bitnet_mlx/cli/api.py
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from typing import List, AsyncGenerator
from ..inference.engine import InjectionEngine

api = FastAPI(title="BitNet-MLX Sovereign API", version="54.0.0")
api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ACTIVE_MODEL_PATH = ""

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="bitnet-local")
    messages: List[Message]
    max_tokens: int = Field(default=512)
    stream: bool = Field(default=False)
    temperature: float = Field(default=0.7)

async def _stream_generator(req: ChatCompletionRequest) -> AsyncGenerator[dict, None]:
    prompt = "\n".join([f"{m.role}: {m.content}" for m in req.messages]) + "\nassistant:"
    
    async for chunk in InjectionEngine.execute_stream(ACTIVE_MODEL_PATH, prompt, req.max_tokens, apply_template=False):
        if "[System]" in chunk: continue
        yield {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "choices": [{"delta": {"content": chunk}}]
        }
    yield {"data": "[DONE]"}

@api.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if not ACTIVE_MODEL_PATH:
        raise HTTPException(status_code=503, detail="Sovereign Logic Engine not mounted.")
        
    if req.stream:
        return EventSourceResponse(_stream_generator(req))
    
    prompt = "\n".join([f"{m.role}: {m.content}" for m in req.messages]) + "\nassistant:"
    full_text = ""
    async for chunk in InjectionEngine.execute_stream(ACTIVE_MODEL_PATH, prompt, req.max_tokens, apply_template=False):
        if "[System]" not in chunk:
            full_text += chunk
            
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": full_text}}]
    }