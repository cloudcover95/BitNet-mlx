# path: BitNet-mlx/benchmarks/quantization_bench.py
#!/usr/bin/env python3
"""
Quantization Benchmarks

Benchmarks for BitNet-mlx 1.58-bit ternary quantization performance.
Run with: python benchmarks/quantization_bench.py
"""

import time
import numpy as np

from src.quantization.model_quantizer import ModelQuantizer


def benchmark_quantization(sizes: list = None, runs: int = 5):
    if sizes is None:
        sizes = [(128, 256), (512, 1024), (1024, 2048), (2048, 4096)]

    print("=== BitNet-mlx Quantization Benchmarks ===\n")
    quantizer = ModelQuantizer(output_dim=128, verbose=False)

    for rows, cols in sizes:
        print(f"Matrix size: {rows}x{cols}")
        dummy = np.random.randn(rows, cols).astype(np.float32)

        times = []
        for _ in range(runs):
            start = time.perf_counter()
            quantizer.quantize_tensor(dummy, name="bench")
            end = time.perf_counter()
            times.append(end - start)

        avg_time = sum(times) / len(times)
        print(f"  Avg time: {avg_time*1000:.2f} ms | Min: {min(times)*1000:.2f} ms | Max: {max(times)*1000:.2f} ms\n")

        quantizer.reset()


if __name__ == "__main__":
    benchmark_quantization()
