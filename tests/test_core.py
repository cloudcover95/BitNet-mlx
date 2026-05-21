import mlx.core as mx
from bitnet_mlx.bitnet_layers import compute_absmean_ternary_ste
from bitnet_mlx.audio_backend import TernaryAudioEncoder
from bitnet_mlx.swarm_rpc import BinaryRPC

def test_ternary_boundaries():
    w = mx.random.normal((128, 128))
    w_q_ste, _, _ = compute_absmean_ternary_ste(w)
    vals = set(mx.unique(mx.round(w_q_ste)).tolist())
    assert vals.issubset({-1.0, 0.0, 1.0})
    print("[+] Ternary boundary test passed")

def test_audio_encoder_kinematics():
    encoder = TernaryAudioEncoder(input_freq=80, hidden_dim=256)
    # Mock Mel-spectrogram: (batch=1, time=500, freq=80)
    dummy_audio = mx.random.normal((1, 500, 80))
    out = encoder(dummy_audio)
    assert out.shape == (1, 500, 256)
    print("[+] Omni-Audio encoder routing passed")

def test_rpc_binary_serialization():
    t_original = mx.random.normal((64, 64))
    payload = BinaryRPC.pack_tensor(t_original)
    t_unpacked = BinaryRPC.unpack_tensor(payload)
    mx.eval(t_unpacked)
    assert mx.array_equal(t_original, t_unpacked).item()
    print("[+] Swarm RPC binary serialization passed")
