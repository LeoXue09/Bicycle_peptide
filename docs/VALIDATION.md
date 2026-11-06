# Validation performed

Validation date: 6 November 2026. These checks cover software behavior and engine interfaces.

| Check | Result |
|---|---|
| Core environment | Python 3.12.7; Biopython 1.88; NumPy 2.5.4. 40 tests discovered, 34 passed and six PyTorch tests skipped. |
| Shared PyTorch environment | Python 3.10.12; PyTorch 2.10.0+cu128, with CUDA hidden. All 40 tests passed, including finite-difference gradients. |
| Synthetic offline demo | Completed geometry screening, covalent task generation and ranking; selected exactly the synthetic_high fixture. |
| RFdiffusion registry/configuration | Process-local potential registration and upstream Hydra `--cfg job` composition passed at the pinned revision. No model diffusion run performed. |
| ProteinMPNN checkpoint smoke | Existing v_48_020 checkpoint loaded on CPU. Two synthetic-backbone samples retained Cys at 1, 7 and 13, with exactly three Cys each; both passed downstream screening. |
| Portable export | Installation/CLI and synthetic demo verified; archive file hashes checked against the manifest. |


CP-Composer inference and covalent HF3 prediction remain unvalidated. Only source/interface preparation was performed for those engines. GPU execution, whole-campaign numerical equivalence, prediction accuracy, synthetic accessibility and binding activity are not established by these checks.

`requirements-structure-py312.lock.txt` records the optional structure/build packages used on Python 3.12; it is not a Python 3.10 environment lock or a PyTorch environment. The core package itself has no third-party runtime dependencies. CI distinguishes core checks from the torch potential checks; the latest remote CI result is available in the repository Actions tab.

Batch task generation and prediction planning are tested with a synthetic mock backend. These tests verify covalent-bond preservation, input hashes, path validation and subprocess output handling; they do not run a structure-prediction model.
