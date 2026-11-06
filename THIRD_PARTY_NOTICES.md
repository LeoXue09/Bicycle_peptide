# Third-party notices

The new framework does not vendor engine implementations or model weights. `engines.lock.json` identifies external repositories. Preserve the license shipped with each exact source revision and check model/database terms separately.

- RFdiffusion: BSD license in the upstream checkout, with University of Washington attribution. The original supplied file contains RFdiffusion-derived classes; its license is retained in `reference/RFDIFFUSION_LICENSE`.
- ProteinMPNN: upstream MIT license; use and cite the original implementation.
- CP-Composer: upstream MIT license, copyright 2024 THUNLP; cite its original paper and source.
- HelixFold3: separate terms in `apps/protein_folding/helixfold3/LICENSE`, including noncommercial and integration restrictions; do not infer permissions from PaddleHelix's top-level license. This package does not relicense that engine.

The supplied `reference/original_potentials.py` preserves the original FGFR2b example, with its inherited RFdiffusion license. It is not executed by the active workflow. Source identity is recorded in `reference/provenance.json`.
