# path: BitNet-mlx/src/inference/ternary_signature.py
#!/usr/bin/env python3
"""
Ternary Signature

Structured container for ternary projection + TDA results.
Designed for direct serialization into .parquet for JuniorMemSys.
"""

import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class TernarySignature:
    """
    Structured ternary analysis result.
    """
    timestamp: float
    input_shape: tuple
    ternary_shape: tuple
    persistence_score: float
    sparsity: float
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.metadata is None:
            d.pop("metadata")
        return d

    @classmethod
    def from_analysis(
        cls,
        analysis_result: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> "TernarySignature":
        proj = analysis_result.get("projection", {})
        pers = analysis_result.get("persistence", {})

        return cls(
            timestamp=time.time(),
            input_shape=proj.get("shape", ()),
            ternary_shape=proj.get("shape", ()),
            persistence_score=pers.get("persistence_score", 0.0),
            sparsity=proj.get("sparsity", 0.0),
            metadata=metadata,
        )
