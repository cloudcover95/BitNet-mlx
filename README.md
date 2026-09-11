# BitNet-mlx

MLX-native 1.58-bit kernels for Apple Silicon (JuniorCloud LLC).
Language + vision quant live here. EOS sheets, ports, and palace do **not**.

## This repo

- Ternary / AbsMean style maps on MLX
- `src/vision_quant.py` — image/text trit path
- Intended consumer: on-device inference on M-series

## Not this repo

JuniorLLM owns:

- custom ports + `ports/layer_mgr.py` (phase → port)
- JuniorTeqp / coolstore / palace (SIS pull does not reseal)
- Draft compile gate, FieldCore port registration
- `scripts/home_sync.py`, `scripts/prove_bitnet.py`

Home index: `cloudcover95/JuniorHome` `docs/JUNIOR_TEQP.md`.

```bash
# property table + layer report (JuniorLLM tree)
PYTHONPATH=../JuniorLLM python ../JuniorLLM/scripts/layer_prod.py
```
