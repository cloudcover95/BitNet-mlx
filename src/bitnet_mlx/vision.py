import mlx.core as mx
import mlx.nn as nn
from .bitnet_layers import DynamicBitLinear
class TernaryMultimodalProjector(nn.Module):
def init(self, vision_dim: int, text_dim: int):
super().init()
self.linear_1 = DynamicBitLinear(vision_dim, text_dim, bias=True)
self.gelu = nn.GELU()
self.linear_2 = DynamicBitLinear(text_dim, text_dim, bias=True)
def call(self, vision_features: mx.array) -> mx.array:
x = self.linear_1(vision_features)
x = self.gelu(x)
return self.linear_2(x)
class TernaryVisionEncoder(nn.Module):
@staticmethod
def patch_vision_tower(vision_model: nn.Module) -> nn.Module:
"""Recursively replace Linear layers with DynamicBitLinear."""
def _recursive_replace(module, name=""):
for child_name, child in list(module.named_children()):
full_name = f"{name}.{child_name}" if name else child_name
if isinstance(child, nn.Linear) and not any(k in full_name.lower() for k in ["embed", "norm"]):
in_d = child.weight.shape[1]
out_d = child.weight.shape[0]
has_bias = hasattr(child, "bias") and child.bias is not None
setattr(module, child_name, DynamicBitLinear(in_d, out_d, bias=has_bias))
else:
_recursive_replace(child, full_name)
_recursive_replace(vision_model)
return vision_model
