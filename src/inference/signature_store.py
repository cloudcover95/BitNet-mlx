# path: BitNet-mlx/src/inference/signature_store.py
#!/usr/bin/env python3
"""
Signature Store

Lightweight Parquet persistence layer for TernarySignature objects.
Designed to feed directly into JuniorMemSys topological memory.
"""

import logging
from pathlib import Path
from typing import List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from .ternary_signature import TernarySignature

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class SignatureStore:
    """
    Simple Parquet-based store for ternary analysis results.
    """

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"SignatureStore initialized at {self.base_path}")

    def save(self, signature: TernarySignature, filename: str = "signatures.parquet"):
        file_path = self.base_path / filename
        table = pa.Table.from_pydict([signature.to_dict()])

        if file_path.exists():
            # Append mode
            existing = pq.read_table(file_path)
            combined = pa.concat_tables([existing, table])
            pq.write_table(combined, file_path, compression="ZSTD")
        else:
            pq.write_table(table, file_path, compression="ZSTD")

        logging.info(f"Saved signature to {file_path}")

    def load_all(self, filename: str = "signatures.parquet") -> List[TernarySignature]:
        file_path = self.base_path / filename
        if not file_path.exists():
            return []

        table = pq.read_table(file_path)
        records = table.to_pylist()

        return [TernarySignature(**r) for r in records]
