import argparse
import sys
from .audit import GrokEvaluator

def main():
    parser = argparse.ArgumentParser(description="BitNet-MLX Sovereign CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    subparsers.add_parser("eval").set_defaults(func=lambda args: GrokEvaluator.run_inference_emulation())
    subparsers.add_parser("convert").set_defaults(func=lambda args: print("[*] Converter engaged."))
    
    args = parser.parse_args()
    if hasattr(args, "func"): args.func(args)
    else: parser.print_help()
