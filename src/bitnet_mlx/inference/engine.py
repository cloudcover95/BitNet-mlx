# src/bitnet_mlx/inference/engine.py
import asyncio
import time
from typing import AsyncGenerator
from mlx_lm import generate_step, load
from ..converter.surgeon import TopologySurgeon
import mlx.core as mx

class InjectionEngine:
    @staticmethod
    async def execute_stream(model_path: str, prompt: str, max_tokens: int = 512, apply_template: bool = True) -> AsyncGenerator[str, None]:
        """KV-cache optimized streaming async generator with TPS metrics."""
        model, tokenizer = load(model_path)
        model = TopologySurgeon.transmute(model)
        
        if apply_template and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            
        prompt_tokens = mx.array(tokenizer.encode(prompt))
        
        start_time = time.perf_counter()
        token_count = 0
        
        for token, _ in zip(generate_step(prompt_tokens, model), range(max_tokens)):
            token_count += 1
            yield tokenizer.decode([token.item()])
            await asyncio.sleep(0.0)
            
        tps = token_count / (time.perf_counter() - start_time)
        yield f"\n\n[System] Inference complete. Throughput: {tps:.2f} tokens/sec."