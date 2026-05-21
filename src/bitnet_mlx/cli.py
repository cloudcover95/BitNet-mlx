import argparse
import logging
import sys
import pandas as pd
from pathlib import Path
from .converter import BitNetCompiler
from .audit import EmulationAuditor
from .engine import InjectionEngine
from .qat import SovereignTrainer

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

def run_c2v_report():
log_file = Path("logs/c2v_telemetry.parquet")
if not log_file.exists(): return logger.error("[-] Root Node Failure: Parquet stream disconnected.")
df = pd.read_parquet(log_file)

global_params = df['total_params'].sum()
global_zeros = df['zeros'].sum()
global_outliers = df['outliers'].sum()

logger.info("\n[*] --- SYSTEM CAPABILITY: HYBRID PARAMETRIC WEIGHTING ---")
logger.info(f"Topologies Processed: {len(df)}")
logger.info(f"Total Logic Gates: {global_params:,}")
logger.info(f"Ternary Zero-State Ratio: {(global_zeros / global_params) * 100:.2f}%")
logger.info(f"FP16 Outlier Mapping: {(global_outliers / global_params) * 100:.4f}%")
logger.info(f"O(N) Variance Preservation: {df['variance_fidelity'].mean()*100:.2f}%")
logger.info("[*] ------------------------------------------------------------")
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
