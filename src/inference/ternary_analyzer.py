# path: BitNet-mlx/src/inference/ternary_analyzer.py
#!/usr/bin/env python3
"""
Ternary Analyzer

High-level interface combining Ternary Projection and Ternary TDA.
Provides a clean entry point for embedding + topological analysis
in the discrete 1.58-bit space.
"""

import logging
from typing import Any, Dict

from .ternary_projection import TernaryProjector
from .ternary_tda import TernaryTDA

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class TernaryAnalyzer:
    """
    Unified interface for ternary projection + combinatorial TDA.
    """

    def __init__(self, output_dim: int = 128, threshold: float = 0.33):
        self.projector = TernaryProjector(output_dim=output_dim, threshold=threshold)
        self.tda = TernaryTDA()
        logging.info("TernaryAnalyzer initialized")

    def analyze(self, x: Any) -> Dict[str, Any]:
        """
        Full pipeline: project input then compute topological signature.
        """
        projection = self.projector.project(x)
        persistence = self.tda.compute_persistence(projection["ternary_embedding"])

        return {
            "projection": projection,
            "persistence": persistence,
            "combined_score": persistence.get("persistence_score", 0.0),
        }

    def analyze_batch(self, batch: list) -> list:
        return [self.analyze(x) for x in batch]
