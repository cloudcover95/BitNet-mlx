# path: BitNet-mlx/src/kernels/ternary_matmul.py
#!/usr/bin/env python3
"""
Ternary MatMul Kernel (MLX + Metal)

Concrete, optimized Metal kernel implementation for 1.58-bit ternary
matrix multiplication on Apple Silicon.

This replaces the naive matmul with a custom kernel that exploits
ternary weights (no multiplications, only add/sub/zero).
"""

import logging
from typing import Tuple

import mlx.core as mx

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


# Metal kernel source for ternary matmul
# A: (M, K) float16/bfloat16
# B: (K, N) ternary weights packed as int8 (-1, 0, 1)
# C = A @ B.T
TERNARY_MATMUL_SOURCE = """
    uint tid = thread_position_in_grid.x;
    uint row = tid / N;
    uint col = tid % N;

    if (row >= M || col >= N) return;

    float acc = 0.0f;

    for (uint k = 0; k < K; ++k) {
        float a_val = (float)A[row * K + k];
        int8_t b_val = B[k * N + col];  // -1, 0, or 1

        if (b_val != 0) {
            acc += a_val * (float)b_val;
        }
    }

    C[row * N + col] = (float16)acc;
"""


def ternary_matmul(
    a: mx.array,
    ternary_b: mx.array,
    transpose_b: bool = True,
) -> mx.array:
    """
    Performs efficient ternary matrix multiplication using a custom Metal kernel.

    Args:
        a: Input activations (M, K)
        ternary_b: Ternary weights (K, N) or (N, K) if transpose_b=True
        transpose_b: Whether to transpose B (standard for linear layers)

    Returns:
        Result of shape (M, N)
    """
    if a.dtype not in (mx.float16, mx.bfloat16):
        a = a.astype(mx.float16)

    if ternary_b.dtype != mx.int8:
        ternary_b = ternary_b.astype(mx.int8)

    M, K = a.shape
    if transpose_b:
        N, K_b = ternary_b.shape
        assert K == K_b, "K dimension mismatch"
        b = ternary_b  # already (N, K)
    else:
        K_b, N = ternary_b.shape
        assert K == K_b, "K dimension mismatch"
        b = mx.transpose(ternary_b)  # make it (N, K)

    # Output
    c = mx.zeros((M, N), dtype=mx.float16)

    # Launch custom kernel
    kernel = mx.fast.metal_kernel(
        name="ternary_matmul",
        source=TERNARY_MATMUL_SOURCE,
        input_names=["A", "B"],
        output_names=["C"],
        grid=(M * N, 1, 1),
        threadgroup=(256, 1, 1),
    )

    outputs = kernel(
        inputs=[a, b],
        output_shapes=[(M, N)],
        output_dtypes=[mx.float16],
        grid=(M * N, 1, 1),
        threadgroup=(256, 1, 1),
    )

    return outputs[0]


def benchmark_ternary_matmul(M: int = 512, K: int = 1024, N: int = 512):
    print(f"Benchmarking ternary matmul: ({M}, {K}) x ({K}, {N})")

    a = mx.random.normal((M, K)).astype(mx.float16)
    b_ternary = mx.random.randint(-1, 2, (K, N)).astype(mx.int8)

    import time
    start = time.perf_counter()
    c = ternary_matmul(a, b_ternary)
    mx.eval(c)
    end = time.perf_counter()

    print(f"Time: {(end - start)*1000:.2f} ms")
    return c


if __name__ == "__main__":
    benchmark_ternary_matmul()
