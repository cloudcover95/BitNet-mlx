import argparse
import logging
import sys
from .audit import EmulationAuditor
from .engine import InjectionEngine

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("BitNet.CLI")

def run_converter():
    logger.info("[*] Converter initialized. See subsequent iterations for robust safetensor compilation.")

def run_emulation_audit():
    if not EmulationAuditor.execute_audit(): 
        sys.exit(1)

def run_inference_stream():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    InjectionEngine.execute_streaming_inference(args.model, args.prompt, args.max_tokens)
