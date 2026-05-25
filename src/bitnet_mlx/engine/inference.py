from mlx_lm import generate, load
from .converter import TopologySurgeon

class InjectionEngine:
    @staticmethod
    def execute(model_path: str, prompt: str, max_tokens: int = 256):
        print(f"[*] Booting Sovereign Logic Engine: {model_path}")
        model, tokenizer = load(model_path)
        
        model = TopologySurgeon.transmute(model)
        
        print(f"[*] Thermal payload secure. Executing streaming logic...\n")
        response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=True)
        return response
