#!/usr/bin/env python3
"""Run upstream inference with a process-local custom potential registration."""
import argparse
from pathlib import Path
import runpy
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--engine", type=Path, required=True)
args, hydra_args = parser.parse_known_args()
engine = args.engine.resolve()
sys.path.insert(0, str(engine))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baidu_bicycle.potential import register
register()
sys.argv = [str(engine / "scripts/run_inference.py"), *hydra_args]
runpy.run_path(sys.argv[0], run_name="__main__")
