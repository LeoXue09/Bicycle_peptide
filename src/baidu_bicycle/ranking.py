"""Joint percentile selection with explicit residue mapping and no silent bypass."""
import csv
import math
from pathlib import Path
from .io import finite, sha256, write_json


def quantile(values, q):
    q = finite(q, "quantile", 0, 1)
    values = sorted(finite(v, "score") for v in values)
    if not values:
        raise ValueError("Cannot rank an empty population")
    x = (len(values) - 1) * q
    low = math.floor(x)
    high = math.ceil(x)
    return values[low] + (x-low) * (values[high]-values[low])


def select_rows(rows, config):
    required = {"task_name", "peptide_chain_plddt", "global_iptm", config["interface_column"],
                "interface_threshold_angstrom"}
    if not rows or any(required - set(row) for row in rows):
        raise ValueError("Empty table or missing required metric/interface columns")
    ids = [row["task_name"] for row in rows]
    if len(set(ids)) != len(ids) or any(not v.strip() for v in ids):
        raise ValueError("task_name must be unique and nonempty within the ranking population")
    if config.get("mapping_verified") is not True:
        raise ValueError("Verify interface numbering and record mapping_verified=true")
    hotspots = {str(v) for v in config["hotspots"]}
    if not hotspots:
        raise ValueError("Hotspots must not be empty")
    mapping = config.get("residue_map")
    # Null map means the explicitly verified identity mapping, not an inferred offset.
    values = []
    for row in rows:
        plddt = finite(row["peptide_chain_plddt"], "peptide pLDDT", 0, 100)
        iptm = finite(row["global_iptm"], "ipTM", 0, 1)
        cutoff = finite(row["interface_threshold_angstrom"], "contact threshold", 0)
        if not math.isclose(cutoff, config.get("contact_threshold_angstrom", 8.0)):
            raise ValueError("Mixed or unexpected contact thresholds")
        residues = {x.strip() for x in row[config["interface_column"]].split(",") if x.strip()}
        if mapping is not None:
            if residues - set(mapping):
                raise ValueError(f"Unmapped interface residues: {sorted(residues-set(mapping))}")
            residues = {str(mapping[r]) for r in residues}
        coverage = len(residues & hotspots) / len(residues) if residues else 0.
        values.append((plddt, iptm, coverage))
    pcut = quantile([v[0] for v in values], config.get("plddt_quantile", .98))
    icut = quantile([v[1] for v in values], config.get("iptm_quantile", .98))
    hcut = finite(config.get("hotspot_fraction", .2), "hotspot fraction", 0, 1)
    audit = []
    for row, (p, i, h) in zip(rows, values):
        audit.append({**row, "hotspot_fraction_recomputed": h,
                      "selected": p >= pcut and i >= icut and h >= hcut})
    return audit, {"population_rows": len(rows), "plddt_threshold": pcut,
                   "iptm_threshold": icut, "hotspot_threshold": hcut,
                   "selected_rows": sum(r["selected"] for r in audit),
                   "quantile_method": "linear interpolation at (n-1)*q; inclusive ties",
                   "denominator": "unique contacted target residues"}


def rank_csv(source, output, config):
    with Path(source).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    audit, stats = select_rows(rows, config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for filename, subset in (("audit.csv", audit), ("selected.csv", [r for r in audit if r["selected"]])):
        with (output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(audit[0]))
            writer.writeheader()
            writer.writerows(subset)
    write_json(output / "selection.json", {**stats, "input_sha256": sha256(source), "config": config})
    return stats
