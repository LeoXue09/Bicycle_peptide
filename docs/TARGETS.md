# Target profiles

The principal applications are PSMA and FAP, the protein target of FAP inhibitors (FAPI). FGFR2b is provided as the original implementation example.

| Profile | Structure | Selected receptor residues | Hotspots |
|---|---|---|---|
| PSMA | 3D7H, chain A | 55–653 and 656–750 | 517, 519, 534, 536, 545, 546, 552, 553 |
| FAP | 6Y0F, chain A | 37–757 | 123, 203, 204, 624, 625, 650, 656, 660, 704, 705, 734 |
| FGFR2b example | 1NUN, chain B | 208–359 | 341–353 |

Each `*.target.json` defines chain, resolved ranges and hotspot IDs. Check the actual input coordinate file against those requested ranges. Preparation fails when a requested residue is absent, so missing regions must be handled explicitly.

`psma.rfdiffusion.example.json` and `fap.rfdiffusion.example.json` are editable starter configurations with a 10-residue binder. Choose peptide length, cysteine positions, guide scale, design count and seed index for each experiment. For 8–10-residue designs, pass `--min-length 8` to geometry and sequence-screening commands. The generic RFdiffusion example retains the longer FGFR2b-style setup.

Target-specific ranking templates use the hotspot numbers above. Set the interface column and residue map for the actual prediction output, then mark the mapping verified. Each target's population is ranked independently unless an explicit study protocol specifies pooling.

Choose linker identity and reactive atom IDs independently of the target name. `linker.example.json` records those experiment-specific inputs. `fgfr2b.hf3_task.example.json` shows the original three-entity task format with LFI. Neither target selection nor RFdiffusion cyclic encoding silently determines the final covalent graph.
