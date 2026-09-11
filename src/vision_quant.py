# path: src/vision_quant.py

"""Ternary maps on vision/feature dicts for on-device quant.

Use: chart overlays, frame embeddings, HUD tensors → {-1,0,1}.
Not an app-tag classifier.
"""

from typing import Any, Dict, List


def ternary_quantize_features(features: Dict[str, float], threshold: float = 0.5) -> Dict[str, float]:
    quantized = {}
    for key, value in features.items():
        if isinstance(value, (int, float)):
            if value > threshold:
                quantized[key] = 1.0
            elif value < -threshold:
                quantized[key] = -1.0
            else:
                quantized[key] = 0.0
        else:
            quantized[key] = value
    return quantized


def batch_ternary_quantize(feature_list: List[Dict[str, float]]) -> List[Dict[str, float]]:
    return [ternary_quantize_features(f) for f in feature_list]


def get_bitnet_vision_stats(quantized_features: Dict[str, float]) -> Dict[str, Any]:
    total = len(quantized_features)
    non_zero = sum(1 for v in quantized_features.values() if v != 0)
    sparsity = 1 - (non_zero / total) if total > 0 else 0
    return {
        "total_features": total,
        "non_zero_features": non_zero,
        "sparsity": sparsity,
        "estimated_speedup_vs_fp16": 2.5 + (sparsity * 1.5),
    }
