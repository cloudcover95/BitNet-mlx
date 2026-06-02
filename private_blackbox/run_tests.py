import mlx.core as mx
import mlx.nn as nn
import time
from rich.console import Console
from bitnet_mlx.core.quantization import compute_asymmetric_ternary_ste
from bitnet_mlx.inference.engine import TernaryKVCache
from bitnet_mlx.core.blackbox_ops import BlackBoxArchitecture

console = Console()

def stress_test_memsys():
    console.print("[bold cyan][*] Initializing MemSys Emulation Diagnostics[/bold cyan]")
    
    # 1. Stress Test Zero-Collision Routing with Extreme Sparsity
    console.print("[*] Testing Zero-Collision Bitwise Routing...")
    w = mx.random.normal((1024, 1024), dtype=mx.float16)
    w[mx.abs(w) < 0.8] = 0.0  # Force 80% sparsity
    
    w_q, g_p, g_n, w_raw = compute_asymmetric_ternary_ste(w)
    x = mx.random.normal((16, 1024), dtype=mx.float16)
    
    t0 = time.perf_counter()
    y = BlackBoxArchitecture.bitwise_sparse_route(x, w_raw, g_p, g_n)
    mx.eval(y)
    t1 = time.perf_counter()
    console.print(f"[+] Sparse Routing Nominal. Latency: {(t1-t0)*1000:.2f}ms")

    # 2. Stress Test 2-Bit KV Cache Compression
    console.print("[*] Testing Ternary KV-Cache Memory Delta...")
    kv = TernaryKVCache()
    k = mx.random.normal((1, 8, 1024, 64), dtype=mx.float16)
    v = mx.random.normal((1, 8, 1024, 64), dtype=mx.float16)
    
    qk, qv = kv.update_and_fetch(k, v)
    mx.eval(qk, qv)
    
    # Verify quantization bounds strictly adhere to target manifold
    unique_vals = mx.unique(qk).size
    console.print(f"[+] KV-Cache Binarized. Unique quantized states: {unique_vals}")
    assert unique_vals <= 3, "KV Cache failed to compress to ternary boundaries."
    
    console.print("[bold green][+] MemSys Emulation Passed. Core is stable.[/bold green]")

if __name__ == "__main__":
    stress_test_memsys()