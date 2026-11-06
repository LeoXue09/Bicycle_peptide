# Engine integration

Source revisions are pinned in `engines.lock.json`. Model weights, databases and environments are installed separately. Each execution profile names the exact engine checkout and Python interpreter.

| Engine | Interface |
|---|---|
| [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) | Macrocycle settings and receptor contigs; `baidu_bicycle.rfdiffusion_entry` registers the custom potential within the inference process. |
| [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) | Native `protein_mpnn_run.py` with a prepared PDB, fixed-position JSONL and Cys omission at nonfixed positions. |
| [CP-Composer](https://github.com/jdp22/CP-Composer_final) | Native MMAP preprocessing followed by `generate.py` and post-generation screening. |
| [HelixFold3](https://github.com/PaddlePaddle/PaddleHelix/tree/dev/apps/protein_folding/helixfold3) | Explicit covalent task export and configurable prediction backend, followed by structure/metric collection. |

## RFdiffusion and ProteinMPNN

Use a compatible, pinned PyTorch environment for these engines. `plan-rfdiffusion` constructs the command without launching it; `run` selects an explicit GPU, checks occupancy and records the invocation. The potential is registered in the process without editing the engine checkout. ProteinMPNN preparation mutates selected residues to CYS before constructing its fixed-position dictionary.

## CP-Composer

Use its pinned upstream `env.yaml` and compatible checkpoint in a separate runtime. The preprocessor uses a 10-Angstrom CB-based pocket after a 6-Angstrom atom-contact check. Input `index.tsv` contains ID, receptor chain and peptide chain, with structures under `pdbs/`. Output is `processed/data.bin` and `processed/index.txt`.

The generated configuration specifies five samples, guidance strength 3, condition 4, `seq_only=true` and `CADS_sampler=false`. Modify these explicitly for an experiment. Inspect the preprocessing output before generation. Screen the emitted candidate structures with their associated sequences.

## Covalent prediction backend

The workflow's task schema, named `bicycle-helm-covalent-v1`, contains HELM peptide entities, a linker entity, a receptor sequence and explicit `covalent_bonds`. `build-hf3-task` writes one task; `build-task-batch` writes tasks from screened candidates. `plan-prediction` converts a task manifest into recorded execution plans using a configured backend command.

A backend must support this schema and retain the covalent bonds. The pinned public PaddleHelix `inference.py` instead requires protein `sequence` inputs and does not consume this HELM/covalent-bond contract. Use a compatible covalent prediction installation for this stage; do not point the profile at that standard CLI. No automatic conversion to an unstapled complex is performed.

The backend profile declares its command as an argument array with `{input_json}` and `{output_dir}` placeholders, required output filenames, repository root and model/dependency input files. Each task gets an independent output directory and plan. Keep a successful backend-specific smoke test with its installation records.

## Output metrics

Use `interface` for heavy-atom contact residues in a declared chain/numbering namespace. `collect` maps backend scalar confidence fields using explicit JSON key paths and verifies the contact record belongs to the same CIF. Public HF3 result files expose fields such as `iptm`, `mean_plddt` and per-chain/atom confidence; select the peptide-specific metric according to the actual output schema.

The HelixFold3 subdirectory has its own [terms](https://github.com/PaddlePaddle/PaddleHelix/blob/dev/apps/protein_folding/helixfold3/LICENSE). Third-party source, weights and databases retain their upstream terms.
