"""PDF-schema task generation, with chemistry made explicit.

The PDF's HELM/entities/covalent_bonds schema is not assumed to be accepted
by the public PaddleHelix inference CLI. See docs/ENGINES.md.
"""
import re
from .io import sequence, integer


def build_task(name, peptide, target, linker, positions, head_to_tail=False, allow_extra_cys=False):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", ".."}:
        raise ValueError("Unsafe task name")
    sequence(peptide)
    sequence(target)
    positions = [integer(p, "cysteine position", 1) for p in positions]
    if len(positions) != 3 or len(set(positions)) != 3:
        raise ValueError("Specify exactly three distinct 1-based cysteine positions")
    if any(p > len(peptide) or peptide[p-1] != "C" for p in positions):
        raise ValueError("Bond positions must refer to peptide cysteines")
    if not allow_extra_cys and peptide.count("C") != 3:
        raise ValueError("Extra cysteines require explicit allow_extra_cys")
    if linker.get("chemistry_verified") is not True:
        raise ValueError("Linker identity and reactive atoms require verified chemistry")
    atoms = linker["reactive_atoms"]
    if len(atoms) != 3 or len(set(atoms)) != 3 or any(not isinstance(a,str) or not a for a in atoms):
        raise ValueError("Need three distinct linker reactive atom IDs")
    if not isinstance(linker.get("ccd"), str) or not linker["ccd"]:
        raise ValueError("Missing linker CCD identity")
    bonds = [{"left_entity": "1-1", "left_position": p, "left_atom_id": "SG",
              "right_entity": "2-1", "right_position": 1, "right_atom_id": atom,
              "bond_type": "covale"} for p, atom in zip(positions, atoms)]
    if head_to_tail:
        bonds.append({"left_entity": "1-1", "left_position": len(peptide), "left_atom_id": "C",
                      "right_entity": "1-1", "right_position": 1, "right_atom_id": "N", "bond_type": "covale"})
    return {"job_name": name, "recycle": 4, "ensemble": 1,
            "entities": [{"type": "protein", "helm": "PEPTIDE1{"+".".join(peptide)+"}$$$$V2.0", "count": 1},
                         {"type": "ligand", "ccd": linker["ccd"], "count": 1},
                         {"type": "protein", "sequence": target, "count": 1}],
            "covalent_bonds": bonds}
