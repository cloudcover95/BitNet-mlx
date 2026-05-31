# src/bitnet_mlx/core/tda_math.py
import mlx.core as mx

class TopologicalManifold:
    """
    Composite persistent homology proxies (Betti-0, Betti-1).
    Zero-SVD strict enforcement mapped directly to MLX primitives.
    """
    @staticmethod
    @mx.compile
    def power_iteration_spectral_norm(matrix: mx.array, num_iters: int = 15) -> mx.array:
        dim = matrix.shape[1]
        v = mx.random.normal((dim, 1), dtype=matrix.dtype)
        v = v / (mx.linalg.norm(v) + 1e-9)
        
        for _ in range(num_iters):
            v = mx.matmul(matrix, v)
            v = v / (mx.linalg.norm(v) + 1e-9)
            
        return mx.squeeze(mx.matmul(v.T, mx.matmul(matrix, v)))

    @staticmethod
    @mx.compile
    def multi_scale_betti_zero(w: mx.array) -> mx.array:
        w_max = mx.max(mx.abs(w)) + 1e-9
        thresholds = mx.array([0.1, 0.25, 0.5, 0.75, 0.9]) * w_max
        densities = [mx.sum(mx.abs(w) > t) / (w.shape[0] * w.shape[1]) for t in thresholds.tolist()]
        return mx.array(densities)

    @staticmethod
    @mx.compile
    def compute_manifold_divergence(w_orig: mx.array, w_recon: mx.array) -> mx.array:
        g_orig = mx.matmul(w_orig.T, w_orig)
        g_recon = mx.matmul(w_recon.T, w_recon)
        
        frob_diff = mx.sqrt(mx.sum(mx.square(g_orig - g_recon)))
        frob_base = mx.sqrt(mx.sum(mx.square(g_orig))) + 1e-9
        frob_penalty = frob_diff / frob_base
        
        spec_orig = TopologicalManifold.power_iteration_spectral_norm(g_orig)
        spec_recon = TopologicalManifold.power_iteration_spectral_norm(g_recon)
        spectral_penalty = mx.abs(spec_orig - spec_recon) / (spec_orig + 1e-9)
        
        b0_orig = TopologicalManifold.multi_scale_betti_zero(w_orig)
        b0_recon = TopologicalManifold.multi_scale_betti_zero(w_recon)
        b0_penalty = mx.mean(mx.abs(b0_orig - b0_recon))
        
        return frob_penalty + (spectral_penalty * 0.2) + (b0_penalty * 0.3)