# path: BitNet-mlx/src/inference/ternary_pipeline.py
#!/usr/bin/env python3
"""
Ternary Pipeline

Unified high-level interface that combines projection, TDA,
signature generation, and optional persistence.

Provides a clean, production-ready entry point for the
entire ternary analysis stack.
"""

import logging
from typing import Any, Dict, Optional

from .ternary_analyzer import TernaryAnalyzer
from .ternary_signature import TernarySignature
from .signature_store import SignatureStore

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class TernaryPipeline:
    """
    End-to-end ternary analysis pipeline.
    """

    def __init__(self, output_dim: int = 128, store_path: Optional[str] = None):
        self.analyzer = TernaryAnalyzer(output_dim=output_dim)
        self.store = SignatureStore(store_path) if store_path else None
        logging.info("TernaryPipeline initialized")

    def run(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Full pipeline: analyze → create signature → optional persist.
        """
        analysis = self.analyzer.analyze(data)
        signature = TernarySignature.from_analysis(analysis, metadata=metadata)

        if self.store:
            self.store.save(signature)

        return {
            "analysis": analysis,
            "signature": signature.to_dict(),
            "persisted": self.store is not None,
        }

    def run_batch(self, batch: list, metadata: Optional[Dict[str, Any]] = None) -> list:
        return [self.run(item, metadata=metadata) for item in batch]
