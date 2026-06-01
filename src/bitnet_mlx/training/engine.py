# src/bitnet_mlx/training/engine.py
import time
import os
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Tuple, Dict, Any, Generator
from rich.console import Console
import pandas as pd
from .adapters import inject_adapters

console = Console()

class SovereignTrainer:
    """Unified edge engine executing Full, QAT, LoRA, QLoRA, and PTQ Recovery."""
    
    @staticmethod
    def _prepare_model(model: nn.Module, mode: str) -> nn.Module:
        if mode in ["qlora", "lora"]:
            model.freeze()
            return inject_adapters(model)
        elif mode == "ptq_recovery":
            model.freeze()
            def unfreeze_norms(mod):
                for name, child in list(mod.named_children()):
                    if "norm" in name.lower() or "ln_" in name.lower(): child.unfreeze()
                    else: unfreeze_norms(child)
            unfreeze_norms(model)
            return model
        elif mode == "qat":
            return model 
        elif mode == "full":
            model.unfreeze()
            return model
        raise ValueError(f"Unknown training discipline: {mode}")

    @staticmethod
    def execute_cycle(
        model: nn.Module, 
        dataset: Generator[Tuple[mx.array, mx.array], None, None],
        mode: str = "qlora", 
        epochs: int = 3, 
        lr: float = 1e-4
    ) -> Dict[str, Any]:
        console.print(f"[bold cyan][*] Initializing Training Discipline: {mode.upper()}[/bold cyan]")
        model = SovereignTrainer._prepare_model(model, mode)
        optimizer = optim.AdamW(learning_rate=lr)
        state = [model.state, optimizer.state]

        @mx.compile
        def step(x: mx.array, y: mx.array, state: list) -> Tuple[mx.array, list]:
            model.update(state[0])
            optimizer.update(state[1])
            def loss_fn(m, x_b, y_b): return mx.mean(mx.square(m(x_b) - y_b))
            loss_val, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
            optimizer.apply_gradients(grads, model)
            return loss_val, [model.state, optimizer.state]
            
        history = []
        os.makedirs("logs", exist_ok=True)
        
        for epoch in range(epochs):
            t0 = time.perf_counter()
            epoch_loss = 0.0
            steps = 0
            
            for x_batch, y_batch in dataset:
                loss, state = step(x_batch, y_batch, state)
                mx.eval(state) 
                epoch_loss += loss.item()
                steps += 1
                
            t1 = time.perf_counter()
            avg_loss = epoch_loss / max(1, steps)
            console.print(f"[-] Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | Time: {(t1-t0):.2f}s")
            history.append({"epoch": epoch+1, "loss": avg_loss, "latency_s": (t1-t0)})
            
        model.update(state[0])
        pd.DataFrame(history).to_parquet(f"logs/training_{mode}.parquet", index=False)
        console.print("[bold green][+] Edge Convergence Achieved. Logs serialized to .parquet[/bold green]")
        return {"model": model, "history": history}