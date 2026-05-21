import logging
import sys
from .audit import GrokEvaluator

logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_converter():
    print("[*] PTQ Converter stub ready (Phase 1)")

def run_emulation_audit():
    if not GrokEvaluator.run_inference_emulation():
        sys.exit(1)

def run_inference_stream():
    print("[*] DynamicBitLinear engine ready")
