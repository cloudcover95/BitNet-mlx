#!/usr/bin/env python3
# path: scripts/check_inference_slice.py
"""Pre-commit guard for additive inference slices (see docs/INFERENCE_SLICE.md).

Inspects the staged git index and fails if the pending slice breaks the
slice rules in docs/INFERENCE_SLICE_CHECKLIST.md:

  * exactly one file staged (override with --max-files N)
  * every staged change is an addition (no modify / delete / rename)
  * nothing under .github/workflows/
  * no model weights (*.safetensors, *.gguf, model.bin, tokenizer.model, ...)

Stdlib only. Read-only: it never stages, commits, or pushes anything.

Usage:
    git add <new-file>
    python3 scripts/check_inference_slice.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from fnmatch import fnmatch

WEIGHT_PATTERNS = (
    "*.safetensors",
    "*.gguf",
    "*.bin",
    "*.npz",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "tokenizer.model",
)
WEIGHT_DIRS = ("model/", "models/")
WORKFLOW_PREFIX = ".github/workflows/"


def staged_changes() -> list[tuple[str, str]]:
    """Return (status, path) pairs for the staged index vs HEAD."""
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-status", "--no-renames"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    changes = []
    for line in out.splitlines():
        if not line.strip():
            continue
        status, _, path = line.partition("\t")
        changes.append((status.strip(), path.strip()))
    return changes


def is_weight(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    if any(fnmatch(name, pat) for pat in WEIGHT_PATTERNS):
        return True
    return any(path.startswith(d) or f"/{d}" in path for d in WEIGHT_DIRS)


def check(changes: list[tuple[str, str]], max_files: int) -> list[str]:
    problems = []
    if not changes:
        problems.append("nothing staged: add exactly one new file")
    if len(changes) > max_files:
        problems.append(f"{len(changes)} files staged; slice allows {max_files}")
    for status, path in changes:
        if status != "A":
            problems.append(f"{path}: status {status} (only new files 'A' allowed)")
        if path.startswith(WORKFLOW_PREFIX):
            problems.append(f"{path}: GitHub Actions workflows are off-limits")
        if is_weight(path):
            problems.append(f"{path}: looks like model weights; never commit them")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-files", type=int, default=1)
    args = parser.parse_args(argv)

    try:
        changes = staged_changes()
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[!] could not read git index: {exc}", file=sys.stderr)
        return 2

    problems = check(changes, args.max_files)
    if problems:
        print("[!] inference slice check FAILED:")
        for p in problems:
            print(f"    - {p}")
        return 1

    for _, path in changes:
        print(f"[ok] new file: {path}")
    print("[SUCCESS] slice is additive-only and ready to commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
