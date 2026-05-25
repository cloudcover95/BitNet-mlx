import argparse
import sys
from ..core.evaluation import GrokEvaluator
from ..engine.converter import TopologySurgeon
from ..engine.inference import InjectionEngine

def main():
    parser = argparse.ArgumentParser(description="BitNet-MLX Sovereign SDK Engine")
    subparsers = parser.add_subparsers(dest="command")
    
    subparsers.add_parser("eval").set_defaults(func=lambda args: _run_eval())
    
    conv = subparsers.add_parser("convert")
    conv.add_argument("--repo", required=True, help="HF Model ID")
    conv.add_argument("--output", required=True, help="Output Path")
    conv.set_defaults(func=lambda args: TopologySurgeon.build_bitnet_manifold(args.repo, args.output))

    chat = subparsers.add_parser("chat")
    chat.add_argument("--model", required=True, help="Local Model Path")
    chat.add_argument("--prompt", required=True, help="Input Context")
    chat.set_defaults(func=lambda args: InjectionEngine.execute(args.model, args.prompt))
    
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
