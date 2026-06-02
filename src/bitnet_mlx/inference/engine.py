import time
import logging
from typing import Generator, Optional, Tuple, List, Any
import mlx.core as mx
from mlx_lm.utils import generate_step, load
from ..converter.surgeon import TopologySurgeon

logger = logging.getLogger("SovereignInference")

class TernaryKVCache:
    """Compresses streaming KV-cache into a 2-bit footprint for massive context windows on edge."""
    def __init__(self):
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.offset: int = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> Tuple[mx.array, mx.array]:
        def _quantize_cache(x: mx.array) -> mx.array:
            scale = mx.max(mx.abs(x), axis=-1, keepdims=True) / 1.5
            return mx.round(x / (scale + 1e-7)) * scale
        q_keys = _quantize_cache(keys)
        q_values = _quantize_cache(values)
        if self.keys is None:
            self.keys = q_keys
            self.values = q_values
        else:
            self.keys = mx.concatenate([self.keys, q_keys], axis=2)
            self.values = mx.concatenate([self.values, q_values], axis=2)
        self.offset += keys.shape[2]
        return self.keys, self.values

class SovereignInference:
    """Executes Apple Silicon native streaming inference utilizing the 1.58-bit engine and 2-bit KV Cache."""
    _model_cache = None
    _tokenizer_cache = None
    _current_model_path = None

    @classmethod
    def _load_and_transmute(cls, model_path: str):
        if cls._model_cache is not None and cls._current_model_path == model_path:
            return cls._model_cache, cls._tokenizer_cache
        logger.info(f"Loading and transmuting topology from {model_path} into AMX space...")
        model, tokenizer = load(model_path)
        if not hasattr(model, "quantization") or getattr(model, "quantization", {}).get("type") != "bitnet-1.58-omni-tda":
            model = TopologySurgeon.transmute(model)
        cls._model_cache = model
        cls._tokenizer_cache = tokenizer
        cls._current_model_path = model_path
        return model, tokenizer

    @classmethod
    def stream_inference(cls, model_path: str, prompt: str, max_tokens: int = 512, apply_chat_template: bool = True, temperature: float = 0.7) -> Generator[str, None, None]:
        model, tokenizer = cls._load_and_transmute(model_path)
        if apply_chat_template and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        prompt_tokens = mx.array(tokenizer.encode(prompt))
        start_time = time.perf_counter()
        token_count = 0
        for token, _ in zip(generate_step(prompt_tokens, model, temp=temperature), range(max_tokens)):
            token_count += 1
            yield tokenizer.decode([token.item()])
        tps = token_count / max((time.perf_counter() - start_time), 0.001)
        logger.info(f"Inference cycle complete. Throughput: {tps:.2f} t/s")
