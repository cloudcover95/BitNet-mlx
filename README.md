# BitNet-mlx

MLX-native **1.58-bit ternary** kernels for Apple Silicon (JuniorCloud LLC).
Weights live in \(\{-1,0,1\}\). This repo is the **quant math + Metal/MLX path**, not a social-tag demo and not the property library.

## Quant

- AbsMean: \(W_q = \mathrm{clip}(\mathrm{round}(W/\mathrm{mean}|W|),-1,1)\)
- Sign-binarize when you explicitly want no zeros
- BitLinear-style scaled dots for on-device layers
- `src/vision_quant.py` — feature maps to trits (charts, frames, overlays), not app tags

## Consumers

| Repo | Role |
|------|------|
| JuniorLLM | Ports, Teqp EOS sheets, coolstore, `ports/layer_mgr.py` (phase → port) |
| JuniorStock | Edge quant trading stack |
| JuniorHome | Index only |

Dense trit states route to AstraReason in JuniorLLM; sparse/coexist stay on FieldCore. That routing is **not** implemented here.

```bash
# EOS + layer report lives next door
PYTHONPATH=../JuniorLLM python ../JuniorLLM/scripts/layer_prod.py
PYTHONPATH=../JuniorLLM python ../JuniorLLM/scripts/prove_bitnet.py
```

MIT / project license as in-tree. Local-first.
