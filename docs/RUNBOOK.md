# PSMA and FAP workflow

Activate the core environment or use `.venv/bin/bicycle`. Run from the repository root and supply absolute paths in experiment configurations. Use a new output directory for every run.

The examples below use PSMA. For FAP, substitute the FAP target, RFdiffusion and ranking profiles.

## 1. Prepare the receptor

```bash
bicycle prepare-target --pdb /data/3D7H.pdb --config configs/psma.target.json --out /work/psma/target
```

Preparation writes `target.pdb` and `target.json`, including the sequence and a map from original PDB residues to contiguous entity positions. Check chain/ranges against the supplied coordinates; adapt resolved ranges explicitly if needed.

## 2. Generate backbones

Copy `configs/profile.example.json` to `host.json` and configure the installed interpreters, engines and weights. Copy `configs/psma.rfdiffusion.example.json` to `experiment.json` and set the prepared receptor path and experiment settings.

```bash
bicycle plan-rfdiffusion --config experiment.json --profile host.json --design-dir /work/psma/backbones --out /work/psma/rfd.plan.json
bicycle run --plan /work/psma/rfd.plan.json --out /work/psma/rfd_execution --gpu 0
```

`design_startnum` supplies the seed index in upstream deterministic mode. Preserve every backbone's `.trb` file and the execution receipt. A fixed length per plan keeps triangle positions well-defined; use separate plans to explore different lengths.

## 3. Screen geometry and design sequences

For a ProteinMPNN run with the 10-residue starter configuration:

```bash
bicycle screen --pdb /work/psma/backbones/design_0.pdb --chain A --min-length 8 --out /work/psma/triangle.json
bicycle prepare-mpnn --pdb /work/psma/backbones/design_0.pdb --chain A --min-length 8 --profile host.json --out /work/psma/mpnn
bicycle run --plan /work/psma/mpnn/plan.json --out /work/psma/mpnn_execution --gpu 0
bicycle screen-mpnn --fasta /work/psma/mpnn/sequences/seqs/backbone.fa --pdb /work/psma/mpnn/backbone.pdb --chain A --prefix psma_design_0 --min-length 8 --out /work/psma/screened
```

The native FASTA row is excluded automatically. Screening writes `screening.json` for all samples and `accepted.json` for accepted samples. ProteinMPNN fixes the three selected CYS residues and excludes additional cysteines.

For the CP-Composer branch:

```bash
bicycle prepare-composer --pdb /work/psma/backbones/design_0.pdb --receptor-chain B --peptide-chain A --profile host.json --out /work/psma/composer
bicycle run --plan /work/psma/composer/01_preprocess.plan.json --out /work/psma/composer_preprocess
bicycle run --plan /work/psma/composer/02_generate.plan.json --out /work/psma/composer_generation --gpu 0
bicycle screen-composer --jsonl /work/psma/composer/generated/results.jsonl --min-length 8 --out /work/psma/screened_cp
```

Use actual output chain IDs. CP-Composer requires its own compatible environment and checkpoint. Four-Cys screening is opt-in through `--allow-four-cys`.

## 4. Build covalent prediction tasks

Copy `configs/linker.example.json` to `linker.json`. Set the intended linker CCD ID and its three reactive atom IDs, and record chemistry verification. The selected Cys positions come from geometric screening. Set `--head-to-tail` only when that additional bond belongs to the intended product.

```bash
bicycle build-task-batch --candidates /work/psma/screened/accepted.json --target /work/psma/target/target.json --linker linker.json --out /work/psma/tasks
```

For one manually specified candidate, use `build-hf3-task --spec candidate.json --out candidate.task.json`. The FGFR2b task example illustrates the format.

## 5. Run a compatible prediction backend

Configure `backend.json` from `configs/prediction_backend.example.json`. It must name an installed backend that accepts the HELM/covalent-bond schema; the example command is a configuration template. See [ENGINES.md](ENGINES.md) for the input contract.

```bash
bicycle plan-prediction --manifest /work/psma/tasks/manifest.json --backend backend.json --out /work/psma/prediction
bicycle run --plan /work/psma/prediction/psma_design_0_0.plan.json --out /work/psma/prediction_run_0 --gpu 0
```

`plans.json` lists every generated plan. Execute each plan with an available explicit GPU. The runner checks expected nonempty output files and preserves command, model-input and output hashes. Keep each successful result CIF and metric JSON with its task ID.

## 6. Collect contacts and confidence

```bash
bicycle interface --cif /work/psma/prediction/outputs/psma_design_0_0/predicted_structure.cif --peptide-chain A --target-chain C --numbering auth --out /work/psma/interface_0.json
bicycle collect --manifest completed_predictions.json --mapping metric_paths.json --out /work/psma/collected
```

A/C above are example CIF chain IDs. Select the actual peptide and receptor chains and the correct author/label numbering namespace.

`completed_predictions.json` is a list of records containing `task_name`, `metrics_json`, `interface_json`, `peptide_helm` and `cif_file`. `metric_paths.json` maps the scalar columns `global_iptm` and `peptide_chain_plddt` to paths in the backend's metric JSON. For example, `["iptm"]` selects a top-level ipTM field; the peptide-chain path depends on the backend output schema. Never substitute global pLDDT for peptide-chain pLDDT.

## 7. Select candidates

Copy `configs/psma.ranking.example.json` to `selection.json`. Record the population and validate the residue mapping before setting `mapping_verified=true`.

```bash
bicycle rank --csv /work/psma/collected/metrics.csv --config selection.json --out /work/psma/selected
```

This writes `selected.csv`, `audit.csv` and the numerical thresholds in `selection.json`. Existing metric CSVs can be ranked directly by configuring their interface column. Preserve visual review, sequence redundancy decisions and synthesis selection separately from the automated score filters.
