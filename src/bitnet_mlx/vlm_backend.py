import mlx.core as mx
import mlx.nn as nn
from .bitnet_layers import DynamicBitLinear

class TernaryMultimodalProjector(nn.Module):
    def __init__(self, vision_dim: int, text_dim: int):
        super().__init__()
        self.linear_1 = DynamicBitLinear(vision_dim, text_dim, bias=True)
        self.gelu = nn.GELU()
        self.linear_2 = DynamicBitLinear(text_dim, text_dim, bias=True)

    def __call__(self, vision_features: mx.array) -> mx.array:
        x = self.linear_1(vision_features)
        x = self.gelu(x)
        return self.linear_2(x)

class VLMCompatibilityLayer:
    """Phase 2: Ingestion manifold for mlx-vlm structural parsing."""
    @staticmethod
    def patch_mlx_vlm_topology(vlm_model: nn.Module) -> nn.Module:
        def _recursive_replace(module, name=""):
            for child_name, child in list(module.named_children()):
                full_name = f"{name}.{child_name}" if name else child_name
                if isinstance(child, nn.Linear) and not any(k in full_name.lower() for k in ["embed", "norm", "head"]):
                    in_d = child.weight.shape[1]
                    out_d = child.weight.shape[0]
                    has_bias = hasattr(child, "bias") and child.bias is not None
                    setattr(module, child_name, DynamicBitLinear(in_d, out_d, bias=has_bias))
                else:
                    _recursive_replace(child, full_name)
        _recursive_replace(vlm_model)
        return vlm_model
