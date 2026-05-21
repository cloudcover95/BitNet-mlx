import mlx.nn as nn
from mlx_lm import load, stream_generate
from .bitnet_layers import DynamicBitLinear
import logging

logger = logging.getLogger("BitNet.Engine")

class InjectionEngine:
@staticmethod
def patch_model_topology(model: nn.Module, activation_bits: int = 8):
def _recursive_replace(module, name=""):
for child_name, child in module.named_children():
full_name = f"{name}.{child_name}" if name else child_name
if isinstance(child, nn.Linear) and not any(k in full_name.lower() for k in ["embed", "lm_head", "gate"]):
in_d = child.weight.shape[1]
out_d = child.weight.shape[0]
has_bias = hasattr(child, "bias") and child.bias is not None
setattr(module, child_name, DynamicBitLinear(in_d, out_d, bias=has_bias, activation_bits=activation_bits))
else:
_recursive_replace(child, full_name)
_recursive_replace(model)
return model

@staticmethod
def execute_streaming_inference(model_path: str, prompt: str, max_tokens: int = 512, bits: int = 8):
    logger.info(f"[*] Booting Aether Sovereign Engine: {model_path} [Precision: a{bits}w1.58 + Outliers]")
    model, tokenizer = load(model_path)
    model = InjectionEngine.patch_model_topology(model, activation_bits=bits)
    model.load_weights(f"{model_path}/model.safetensors", strict=False)
    
    logger.info("[*] Hybrid Outlier matrices bound. Executing token stream:\n")
    for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
        print(response.text, end="", flush=True)
    print("\n\n[*] Stream cycle complete.")
