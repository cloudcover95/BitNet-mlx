# path: BitNet-mlx/src/quantization/gemma_ternary_converter.py
#!/usr/bin/env python3
"""
Gemma 4B Ternary Converter (Updated)

Uses the new robust ModelQuantizer engine for rigorous 1.58-bit
ternary conversion of Google Gemma models.
"""

import logging
from typing import Any, Dict, Optional

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from .model_quantizer import ModelQuantizer
    HAS_QUANTIZER = True
except ImportError:
    HAS_QUANTIZER = False

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class GemmaTernaryConverter:
    """
    Converts Gemma models to 1.58-bit ternary using the improved BitNet-mlx engine.
    """

    def __init__(self, model_name: str = "google/gemma-2-2b", output_dim: int = 128):
        self.model_name = model_name

        if not HAS_TRANSFORMERS:
            raise RuntimeError("transformers is required")

        if not HAS_QUANTIZER:
            raise RuntimeError("ModelQuantizer is required")

        logging.info(f"Loading Gemma model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

        self.quantizer = ModelQuantizer(output_dim=output_dim, verbose=True)
        logging.info("GemmaTernaryConverter ready")

    def convert_model(self) -> Dict[str, Any]:
        logging.info("Starting full ternary quantization of Gemma model...")
        summary = self.quantizer.quantize_model(self.model)
        return summary

    def get_report(self):
        return self.quantizer.get_report()
