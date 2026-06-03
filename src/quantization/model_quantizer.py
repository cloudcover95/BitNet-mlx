# path: BitNet-mlx/src/quantization/model_quantizer.py
#!/usr/bin/env python3
"""
Model Quantizer

Production-grade engine for applying BitNet-mlx 1.58-bit ternary
quantization to any transformer model (Gemma, Llama, Mistral, etc.).

This is the core engine that the Gemma converter and future converters use.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .ternary_pipeline import TernaryPipeline

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class ModelQuantizer:
    """
    Core engine for ternary quantization of transformer models.
    """

    def __init__(self, output_dim: int = 128, verbose: bool = True):
        self.pipeline = TernaryPipeline(output_dim=output_dim)
        self.verbose = verbose
        self.quantization_report: List[Dict[str, Any]] = []
        logging.info("ModelQuantizer initialized")

    def quantize_tensor(self, tensor: Any, name: str = "unnamed") -> Dict[str, Any]:
        """
        Quantize a single tensor (weight matrix) using the ternary pipeline.
        """
        if hasattr(tensor, "detach"):
            arr = tensor.detach().cpu().numpy()
        else:
            arr = np.array(tensor)

        result = self.pipeline.run(arr)

        report = {
            "name": name,
            "original_shape": arr.shape,
            "ternary_dim": self.pipeline.analyzer.projector.output_dim,
            "persistence_score": result.get("signature", {}).get("persistence_score", 0.0),
            "sparsity": result.get("projection", {}).get("sparsity", 0.0),
        }

        self.quantization_report.append(report)

        if self.verbose:
            logging.info(f"Quantized {name} | shape={arr.shape} | persistence={report['persistence_score']:.4f}")

        return result

    def quantize_model(self, model: Any, target_modules: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Quantize an entire model layer by layer.
        """
        if target_modules is None:
            # Default: quantize common weight matrices
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens"]

        total_layers = 0
        quantized_layers = 0

        for name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{name}.{param_name}" if name else param_name

                # Check if this layer should be quantized
                should_quantize = any(target in full_name for target in target_modules)

                if should_quantize and param.dim() >= 2:  # Only quantize 2D+ weight matrices
                    try:
                        self.quantize_tensor(param, name=full_name)
                        quantized_layers += 1
                    except Exception as e:
                        logging.error(f"Failed to quantize {full_name}: {e}")

                total_layers += 1

        summary = {
            "total_parameters_scanned": total_layers,
            "layers_quantized": quantized_layers,
            "quantization_report": self.quantization_report,
            "average_persistence": np.mean([r["persistence_score"] for r in self.quantization_report]) if self.quantization_report else 0,
        }

        logging.info(f"Model quantization complete. Quantized {quantized_layers}/{total_layers} layers.")
        return summary

    def get_report(self) -> List[Dict[str, Any]]:
        return self.quantization_report

    def reset(self):
        self.quantization_report = []
        self.pipeline = TernaryPipeline(output_dim=self.pipeline.analyzer.projector.output_dim)
