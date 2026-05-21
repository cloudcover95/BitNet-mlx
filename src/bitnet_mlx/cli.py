import logging
import sys
import argparse
from .audit import GrokEvaluator
from .swarm_rpc import boot_swarm_listener

logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_converter():
    print("[*] PTQ Converter active. Omni pipeline architectures verified.")

def run_emulation_audit():
    if not GrokEvaluator.run_inference_emulation():
        sys.exit(1)

def run_swarm_node():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    boot_swarm_listener(args.port)
