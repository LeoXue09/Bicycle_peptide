# Installation

## Core tools

Use Python 3.10 or newer and an isolated environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[structure]'
.venv/bin/bicycle --help
```

The core package has no mandatory third-party dependencies. The `structure` extra installs Biopython for mmCIF analysis. `requirements-structure-py312.lock.txt` records the optional structure/build dependencies tested with Python 3.12.

## Model engines

```bash
.venv/bin/python scripts/01_fetch_engines.py --out /ssd/path/engines
```

The fetcher uses `engines.lock.json` and excludes model weights and large datasets. Follow each pinned engine's installation and checkpoint instructions. Use separate interpreters where dependency requirements differ; do not install every engine into the lightweight core environment.

Copy `configs/profile.example.json` to `host.json` and set absolute paths for RFdiffusion, its checkpoint, ProteinMPNN, its weight directory, and optionally CP-Composer and its checkpoint. Keep `host.json` local. For covalent prediction, create `backend.json` from `configs/prediction_backend.example.json` and [ENGINES.md](ENGINES.md).

## Checks

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/00_demo.py --out demo_output
```

Run the same tests with the RFdiffusion interpreter and `PYTHONPATH=src` to include the PyTorch potential tests. Select a free physical GPU explicitly for model execution; `bicycle run --gpu 0` records occupancy before launch. Tests and the offline demo do not need a GPU.
