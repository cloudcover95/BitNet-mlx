import argparse
from .converter import BitNetCompiler

def run_converter():
    parser = argparse.ArgumentParser(description="BitNet-MLX Standalone PTQ Compiler")
    parser.add_argument("--input", type=str, required=True, help="HF Repo ID or Local Path")
    parser.add_argument("--output", type=str, required=True, help="Output target directory")
    args = parser.parse_args()
    
    compiler = BitNetCompiler()
    compiler.convert_to_ternary(args.input, args.output)
