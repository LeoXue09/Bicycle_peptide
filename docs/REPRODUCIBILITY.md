# Reproducibility conventions

The manuscript application concerns PSMA and FAP/FAPI. FGFR2b is the original implementation example and has an independent target profile. Record all target-specific settings with each experiment.

## Inputs and indexing

Preserve prepared target coordinates, source structure identity, chain selection, resolved residue ranges and sequence. `prepare-target` exports a residue map between crystallographic IDs and contiguous entity positions. Use that map, or an explicitly validated sequence alignment, when interpreting predicted contacts.

Geometry uses zero-based residue positions and Angstrom distances. Fixed positions supplied to ProteinMPNN and covalent task positions are one-based. Sequence and coordinate lengths must agree. Allowed cysteine counts, minimum sequence length, selected triangle and final covalent topology are explicit settings.

## Geometric guidance

For three selected CA distances `d`, targets `d*`, Huber delta 2 and weight `k`, the energy is:

`E = 10 k tanh(sum(Huber(d - d*)) / 10)`.

The active `bicycle_triangle_v1.compute()` returns `-E`, consistent with RFdiffusion maximizing its guiding potentials. It uses epsilon `1e-6`, distance clipping `[1, 50]`, Huber delta 2 and saturation 10. Invalid indices and nonfinite coordinates raise errors. Unit tests check the gradient numerically and verify that an ascent step lowers the energy.

The supplied FGFR2b implementation is preserved separately in `reference/`. Experiment records must name the implementation and its parameters. Geometry screening is a separate step with a strict default interval `6 < d < 10`; it does not certify chemical feasibility.

## Sequence design and prediction

ProteinMPNN fixes three actual CYS residues in the input structure and excludes extra Cys from sampled positions. The native FASTA row is excluded from post-design sample screening. CP-Composer candidates are checked against their emitted structures. Four-Cys mode is available only when explicitly selected.

Covalent prediction records the linker CCD identifier, reactive atom IDs, three selected cysteines and any head-to-tail bond. Backend capability and schema are declared in the host profile. Keep the backend source/model versions and a successful representative input/output test with the run.

## Selection population

Define the target, design branch and prediction population before computing quantiles. Selection uses linear-interpolated quantiles with inclusive ties. The hotspot fraction denominator is the set of unique contacted target residues. Missing metrics, duplicate IDs, mismatched contact thresholds or unverified mappings cause errors rather than a silently weakened selection rule.

## Execution records

`bicycle run` records commands, input and output SHA256 hashes, engine commits/status, GPU selection, elapsed time and failures. Use new output directories and retain original backbone `.trb` mappings. Keep exact model hashes and environment package records. Example configurations are starting points for new runs; they are not records of a previously executed experiment.

See [VALIDATION.md](VALIDATION.md) for the checks performed on this software release. Backend integration tests and a full model experiment provide different evidence and should be recorded separately.
