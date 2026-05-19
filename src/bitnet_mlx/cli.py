import argparse
from .converter import BitNetCompiler

def run_converter():
    parser = argparse.ArgumentParser(description="BitNet-MLX Standalone Enterprise Compiler")
    parser.add_argument("--input", type=str, required=True, help="HF Repo ID or Local Manifold Path")
    parser.add_argument("--output", type=str, required=True, help="Output Target Directory")
    args = parser.parse_args()
    
    compiler = BitNetCompiler()
    compiler.compile_manifold(args.input, args.output)
