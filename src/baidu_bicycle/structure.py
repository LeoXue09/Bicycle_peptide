"""Explicit PDB residue identities and optional Biopython mmCIF reader."""
from pathlib import Path
from math import dist, isfinite
from .io import write_json, sha256

RESIDUES = dict(zip(
    "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(),
    "ACDEFGHIKLMNPQRSTVWY"))


def pdb_residues(path, chain):
    records = {}
    models = 0
    for line in Path(path).read_text().splitlines():
        if line.startswith("MODEL"):
            models += 1
            if models > 1:
                raise ValueError("Multiple PDB models: select one explicitly")
        if not line.startswith("ATOM  ") or line[21:22] != chain:
            continue
        if line[16:17] not in (" ", "A"):
            continue
        key = (int(line[22:26]), line[26:27].strip())
        name = line[17:20].strip()
        if name not in RESIDUES:
            raise ValueError(f"Unsupported residue {chain}{key}: {name}")
        res = records.setdefault(key, {"chain": chain, "number": key[0], "icode": key[1],
                                       "aa": RESIDUES[name], "atoms": {}, "lines": []})
        if res["aa"] != RESIDUES[name]:
            raise ValueError("Inconsistent alternate residue identities")
        atom = line[12:16].strip()
        if atom in res["atoms"]:
            raise ValueError(f"Duplicate atom/altloc {chain}{key} {atom}; prepare a single conformer")
        xyz = [float(line[i:i+8]) for i in (30,38,46)]
        if not all(isfinite(v) for v in xyz):
            raise ValueError("Nonfinite PDB coordinates")
        res["atoms"][atom] = xyz
        res["lines"].append(line)
    if not records:
        raise ValueError(f"No canonical ATOM residues in chain {chain}")
    if any("CA" not in r["atoms"] for r in records.values()):
        raise ValueError("Missing CA atoms; repair structure explicitly")
    return list(records.values())


def prepare_target(source, output, chain, ranges, hotspots):
    residues = pdb_residues(source, chain)
    if not ranges or any(len(r) != 2 or r[0] > r[1] for r in ranges):
        raise ValueError("Require nonempty inclusive residue ranges")
    expected = {n for start, end in ranges for n in range(start, end+1)}
    chosen = [r for r in residues if r["number"] in expected]
    actual = {r["number"] for r in chosen}
    if expected - actual:
        raise ValueError(f"Unresolved requested residues: {sorted(expected-actual)}; supply resolved ranges")
    if any(r["icode"] for r in chosen):
        raise ValueError("Insertion codes require an explicit upstream contig mapping")
    if set(hotspots) - actual:
        raise ValueError("Hotspots absent from selected target")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "target.pdb").write_text("\n".join(line for r in chosen for line in r["lines"]) + "\nTER\nEND\n")
    mapping = [{"entity_position_1based": i+1, "pdb_chain": chain, "pdb_residue": str(r["number"]),
                "amino_acid": r["aa"]} for i, r in enumerate(chosen)]
    write_json(output / "target.json", {"source_sha256": sha256(source),
               "sequence": "".join(r["aa"] for r in chosen), "residue_mapping": mapping,
               "hotspots_pdb": [str(n) for n in hotspots],
               "hotspots_entity_1based": [i+1 for i,r in enumerate(chosen) if r["number"] in hotspots]})
    return output / "target.pdb"


def cif_interface(path, peptide_chain, target_chain, cutoff=8.0, numbering="auth"):
    """Unique target residues within cutoff of any peptide heavy atom.

    Chain IDs and numbering namespace must be supplied, never inferred from order.
    B factors are not interpreted as pLDDT here.
    """
    from Bio.PDB import MMCIFParser
    if numbering not in {"auth", "label"} or not isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Invalid numbering namespace or contact threshold")
    if peptide_chain == target_chain:
        raise ValueError("Peptide and target chains must differ")
    parser = MMCIFParser(QUIET=True, auth_chains=numbering == "auth", auth_residues=numbering == "auth")
    structure = parser.get_structure("complex", str(path))
    if len(structure) != 1:
        raise ValueError("Require a single structure model")
    model = structure[0]
    peptide, target = model[peptide_chain], model[target_chain]
    def heavy(residue):
        return [a.coord for a in residue if a.element.upper() not in {"H", "D"}]
    peptide_atoms = [a for r in peptide if r.id[0] == " " for a in heavy(r)]
    if not peptide_atoms:
        raise ValueError("No peptide atoms")
    contacts = []
    for residue in target:
        if residue.id[0] != " ":
            continue
        if any(dist(a,b) <= cutoff for a in heavy(residue) for b in peptide_atoms):
            contacts.append(str(residue.id[1]) + residue.id[2].strip())
    return {"target_interface_residues": contacts, "numbering": numbering,
            "peptide_chain": peptide_chain, "target_chain": target_chain,
            "interface_threshold_angstrom": cutoff, "structure_sha256": sha256(path)}
