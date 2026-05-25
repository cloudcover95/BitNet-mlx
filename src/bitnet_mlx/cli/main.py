import argparse
import sys
from ..core.evaluation import GrokEvaluator
from .mcp_server import start_server

def main():
    parser = argparse.ArgumentParser(description="BitNet-MLX Sovereign Matrix CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    subparsers.add_parser("eval").set_defaults(func=lambda args: _run_eval())
    subparsers.add_parser("mcp").set_defaults(func=lambda args: start_server())
    
    args = parser.parse_args()
    if hasattr(args, "func"): 
        args.func(args)
    else: 
        parser.print_help()

def _run_eval():
    if not GrokEvaluator.run_inference_emulation():
        sys.exit(1)

if __name__ == "__main__":
    main()
