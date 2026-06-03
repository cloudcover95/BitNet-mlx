# path: BitNet-mlx/src/inference/bitnet_bridge.py
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import mlx.core as mx
    import mlx_lm
    HAS_MLX_LM = True
except ImportError:
    HAS_MLX_LM = False


class BitNetCognitiveBridge:
    """
    V1.1: Sovereign LLM Reasoning Node with efficient daily logging.

    Uses daily append-only markdown files instead of one file per debate.
    Dramatically reduces SSD metadata writes while remaining Obsidian-friendly.
    """

    def __init__(self, model_path: str = "mlx-community/bitnet-1.58b-mlx", vault_root: Optional[str] = None):
        if vault_root:
            self.vault_path = Path(vault_root)
        else:
            self.vault_path = Path.home() / "JuniorCloud" / "juniorstock" / "vault" / "obsidian_logs"

        self.vault_path.mkdir(parents=True, exist_ok=True)
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

        if HAS_MLX_LM:
            try:
                self.model, self.tokenizer = mlx_lm.load(model_path)
                self.active = True
            except Exception:
                self.active = False
        else:
            self.active = False

    def generate_debate_log(self, ticker: str, consensus_data: Dict[str, Any], context: Dict[str, Any]) -> str:
        reasoning = "N/A - MLX_LM offline."

        if self.active and self.model and self.tokenizer:
            prompt = (
                f"<|im_start|>system\nYou are an elite quantitative analyst for JuniorCloud LLC. "
                f"Review the deterministic swarm consensus and output a strict 3-bullet executive summary.\n"
                f"<|im_end|>\n<|im_start|>user\nTicker: {ticker}\nAction: {consensus_data.get('action_proposal')}\n"
                f"Score: {consensus_data.get('consensus_score')}\nContext: {context}<|im_end|>\n<|im_start|>assistant\n"
            )
            try:
                reasoning = mlx_lm.generate(
                    self.model, self.tokenizer, prompt=prompt, max_tokens=120, verbose=False
                )
            except Exception:
                pass

        # Use daily append-only log instead of one file per debate
        date_str = time.strftime("%Y-%m-%d")
        daily_file = self.vault_path / f"debates_{date_str}.md"

        entry = f"""\n---\n**{time.strftime('%H:%M:%S')}** | {ticker} | {consensus_data.get('action_proposal')}\n\n{reasoning.strip()}\n\n```json\n{consensus_data}\n```\n\n"""

        try:
            with open(daily_file, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception:
            pass

        return reasoning
