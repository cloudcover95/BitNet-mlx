import mlx.core as mx
import mlx.nn as nn
from .bitnet_layers import DynamicBitLinear

class TernaryAudioEncoder(nn.Module):
    """Phase 3: High-density audio feature encoding mapping spectrogram arrays into ternary manifolds."""
    def __init__(self, input_freq: int, hidden_dim: int):
        super().__init__()
        # Audio specific gamma scaling applied automatically inside DynamicBitLinear
        self.conv_1d_proxy = DynamicBitLinear(input_freq, hidden_dim, bias=True, modality="audio")
        self.silu = nn.SiLU()
        self.proj = DynamicBitLinear(hidden_dim, hidden_dim, bias=False, modality="audio")

    def __call__(self, mel_spectrogram: mx.array) -> mx.array:
        """Expects shape: (batch, time_steps, frequencies)"""
        x = self.conv_1d_proxy(mel_spectrogram)
        x = self.silu(x)
        return self.proj(x)
