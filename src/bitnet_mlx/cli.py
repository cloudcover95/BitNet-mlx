import argparse
import logging
import sys
import json
import uvicorn
import pandas as pd
from pathlib import Path
from .converter import BitNetCompiler
from .audit import EmulationAuditor
from .engine import InjectionEngine
from .qat import SovereignTrainer
from .swarm import app

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("BitNet.CLI")

def run_converter():
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--auto-tune", action="store_true", default=True)
args = parser.parse_args()
BitNetCompiler(enable_autotune=args.auto_tune).compile_manifold(args.input, args.output)

def run_emulation_audit():
parser = argparse.ArgumentParser()
parser.add_argument("--export-json", action="store_true")
args = parser.parse_args()
if not EmulationAuditor().execute_audit(export_json=args.export_json): sys.exit(1)

def run_metadata_inspector():
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
args = parser.parse_args()
model_path = Path(args.model)
for shard in model_path.glob(".safetensors"):
with open(shard, 'rb') as f:
header_len = int.from_bytes(f.read(8), 'little')
header = json.loads(f.read(header_len).decode('utf-8'))
meta = header.get("metadata", {})
if "juniorcloud-bitnet-v21-ethereal" in meta.get("format", ""):
tp = int(meta.get("c2v_total_params", 1))
logger.info(f"\n[] --- SYSTEM CAPABILITY: ETHEREAL SOVEREIGN MANIFEST ---")
logger.info(f"Topological Shard: {shard.name}")
logger.info(f"Zero-State Ratio: {(int(meta.get('c2v_zeros', 0)) / tp) * 100:.2f}%")
logger.info(f"FP16 Outliers: {(int(meta.get('c2v_outliers', 0)) / tp) * 100:.4f}%")
return
logger.error("[-] Payload missing. Not an Ethereal array.")

def run_c2v_report():
log_file = Path("logs/c2v_telemetry.parquet")
if not log_file.exists(): return logger.error("[-] Root Node Failure: Parquet missing.")
df = pd.read_parquet(log_file)
logger.info(f"\n[*] --- SYSTEM CAPABILITY: C2V PARAMETER MANIFEST ---")
logger.info(f"Global O(N) Variance Preservation: {df['variance_fidelity'].mean()100:.2f}%")
logger.info("[] ------------------------------------------------------------")

def run_swarm_node():
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=9000)
args = parser.parse_args()
logger.info(f"[*] Booting Swarm RPC Node on port {args.port}...")
uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")

def run_inference_stream():
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--prompt", required=True)
parser.add_argument("--max-tokens", type=int, default=512)
parser.add_argument("--activation-bits", type=int, default=8, choices=[4, 8])
args = parser.parse_args()
InjectionEngine.execute_streaming_inference(args.model, args.prompt, args.max_tokens, args.activation_bits)

def run_qat_pipeline():
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--epochs", type=int, default=1)
args = parser.parse_args()
trainer = SovereignTrainer(args.model)
trainer.execute_training_loop(args.dataset, args.epochs)
