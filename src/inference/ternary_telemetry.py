# path: BitNet-mlx/src/inference/ternary_telemetry.py
#!/usr/bin/env python3
"""
Ternary Telemetry Adapter

Convenience layer to ingest common data formats (list, numpy, dict)
and run them through the full TernaryAnalyzer pipeline.
"""

import logging
from typing import Any, Dict, List, Union

import numpy as np

from .ternary_analyzer import TernaryAnalyzer

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class TernaryTelemetry:
    """
    Easy ingestion layer for telemetry and sensor data.
    """

    def __init__(self, output_dim: int = 128):
        self.analyzer = TernaryAnalyzer(output_dim=output_dim)
        logging.info("TernaryTelemetry initialized")

    def from_list(self, data: List[float]) -> Dict[str, Any]:
        arr = np.array(data, dtype=np.float32)
        return self.analyzer.analyze(arr)

    def from_numpy(self, arr: np.ndarray) -> Dict[str, Any]:
        return self.analyzer.analyze(arr)

    def from_dict(self, data: Dict[str, float]) -> Dict[str, Any]:
        arr = np.array(list(data.values()), dtype=np.float32)
        result = self.analyzer.analyze(arr)
        result["keys"] = list(data.keys())
        return result

    def process_batch(self, batch: List[Any]) -> List[Dict[str, Any]]:
        results = []
        for item in batch:
            if isinstance(item, list):
                results.append(self.from_list(item))
            elif isinstance(item, np.ndarray):
                results.append(self.from_numpy(item))
            elif isinstance(item, dict):
                results.append(self.from_dict(item))
            else:
                logging.warning(f"Unsupported type: {type(item)}")
        return results
