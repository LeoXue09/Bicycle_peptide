# Baidu Bicycle

An RFdiffusion-based workflow for designing chemically stapled bicyclic peptide binders to **PSMA and fibroblast activation protein (FAP)**, with applications to FAP inhibitor (FAPI) development.

The toolkit covers target preparation, backbone generation, cysteine geometry, ProteinMPNN and CP-Composer sequence design, covalent prediction inputs, structural analysis and candidate selection. The original FGFR2b implementation is retained as an additional example; target settings are configured independently.

## Install and run the demo

```bash
git clone https://github.com/LeoXue09/Baidu-Bicycle.git
cd Baidu-Bicycle
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[structure]'
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/00_demo.py --out demo_output
```

Python 3.10 or newer is required. The demo runs offline with synthetic structures and scores. GPU design uses separately installed engines and model parameters; see [installation](docs/INSTALLATION.md).

## Workflow

| Stage | Command | Output |
|---|---|---|
| Target preparation | `bicycle prepare-target` | Receptor PDB, sequence and residue mapping |
| Backbone generation | `bicycle plan-rfdiffusion`, `bicycle run` | RFdiffusion backbones and execution records |
| Geometry screening | `bicycle screen` | Cysteine triangle positions and distances |
| ProteinMPNN design | `bicycle prepare-mpnn`, `bicycle run`, `bicycle screen-mpnn` | Sequences with three fixed cysteines |
| CP-Composer design | `bicycle prepare-composer`, `bicycle run`, `bicycle screen-composer` | Pocket-conditioned sequence candidates |
| Covalent task generation | `bicycle build-hf3-task`, `bicycle build-task-batch` | Peptide, linker, receptor and explicit covalent bonds |
| Prediction integration | `bicycle plan-prediction`, `bicycle run` | Recorded calls to a configured covalent prediction backend |
| Interface analysis | `bicycle interface`, `bicycle collect` | Contact residues and confidence metrics |
| Candidate selection | `bicycle rank` | Selected candidates, full decision table and thresholds |

Start with the [PSMA/FAP runbook](docs/RUNBOOK.md). [Target profiles](docs/TARGETS.md) describe the PSMA, FAP and FGFR2b examples. Every command supports `--help`.

## Design method

RFdiffusion generates receptor-conditioned peptide backbones. A differentiable Huber/tanh potential guides three selected C-alpha positions toward a compatible triangle. ProteinMPNN fixes cysteines at selected positions and excludes additional cysteines; CP-Composer candidates undergo post-generation geometric screening. Accepted sequences are represented with explicit linker bonds for complex prediction.

Candidate selection combines peptide-chain pLDDT and interface pTM quantiles with the fraction of contacted target residues overlapping the specified hotspot region. The workflow records input hashes, engine commits, commands, model files, timing and output hashes.

The toolkit uses Angstroms for distances. Geometry indices are zero-based; ProteinMPNN fixed positions and covalent task positions are one-based. Residue maps keep crystallographic numbering separate from contiguous prediction-entity numbering.

## External models

Pinned source references are in `engines.lock.json`; download them with:

```bash
.venv/bin/python scripts/01_fetch_engines.py --out /ssd/path/engines
```

RFdiffusion and ProteinMPNN plans use their native entry points. CP-Composer has its own runtime and checkpoint. Covalent prediction uses the HELM/covalent-bond backend contract described in [engine integration](docs/ENGINES.md); the standard public PaddleHelix CLI is a different input interface. Engine weights and reference databases are installed separately under their upstream terms.

## Validation and provenance

The release includes unit and integration tests, GitHub Actions, an offline demo and a CPU ProteinMPNN checkpoint smoke test. See [validation](docs/VALIDATION.md) and [reproducibility conventions](docs/REPRODUCIBILITY.md) for the tested scope. Synthetic fixtures are software tests, not experimental candidates.

`reference/original_potentials.py` preserves the supplied FGFR2b implementation. The active `bicycle_triangle_v1` implementation uses RFdiffusion's gradient-ascent convention and is independently tested. The reference module is not imported by the workflow.

## License and citation

The framework is distributed under the [MIT license](LICENSE). External engines and the RFdiffusion-derived reference file retain their own licenses; see [third-party notices](THIRD_PARTY_NOTICES.md). Citation metadata is available in [CITATION.cff](CITATION.cff). Please also cite the original RFdiffusion, ProteinMPNN, CP-Composer and prediction-engine publications used in your experiment.
