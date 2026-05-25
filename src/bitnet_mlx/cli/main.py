import argparse
import sys
from ..core.evaluation import GrokEvaluator
from ..engine.converter import SovereignConverter
from ..engine.inference import InjectionEngine
from .mcp_server import start_server

def main():
    parser = argparse.ArgumentParser(description="BitNet-MLX Sovereign Execution Suite")
    subparsers = parser.add_subparsers(dest="command")
    
    subparsers.add_parser("eval").set_defaults(func=lambda args: _run_eval())
    subparsers.add_parser("mcp").set_defaults(func=lambda args: start_server())
    
    conv_parser = subparsers.add_parser("convert")
    conv_parser.add_argument("--repo", required=True, help="HF Model ID")
    conv_parser.add_argument("--output", required=True, help="Output Path")
    conv_parser.set_defaults(func=lambda args: SovereignConverter.build_bitnet_manifold(args.repo, args.output))

    chat_parser = subparsers.add_parser("chat")
    chat_parser.add_argument("--model", required=True, help="Local Model Path")
    chat_parser.add_argument("--prompt", required=True, help="Input Context")
    chat_parser.set_defaults(func=lambda args: InjectionEngine.execute(args.model, args.prompt))
    
    args = parser.parse_args()
    if hasattr(args, "func"): args.func(args)
    else: parser.print_help()

def _run_eval():
    if not GrokEvaluator.run_inference_emulation():
        print("[-] FATAL: Fidelity breached.")
        sys.exit(1)

if __name__ == "__main__":
    main()
