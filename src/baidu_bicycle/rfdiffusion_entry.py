"""Module entrypoint so a shared engine remains untouched."""
import argparse
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, type=Path)
    args, overrides = parser.parse_known_args()
    engine = args.engine.resolve()
    sys.path.insert(0, str(engine))
    from .potential import register
    register()
    sys.argv = [str(engine/"scripts/run_inference.py"), *overrides]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
