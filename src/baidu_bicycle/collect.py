"""Collect explicitly mapped engine metrics; never substitute global for peptide pLDDT."""
import csv
from pathlib import Path
from .io import read_json, finite, sha256, write_json


def lookup(record, path):
    for key in path:
        record = record[key]
    return record


def collect(manifest, mapping, output):
    """manifest records: task_name, metrics_json, interface_json, peptide_helm, cif_file.

    mapping gives key paths (lists of dict keys/list indices) for the scalar scores.
    The interface JSON is produced by `bicycle interface` from the same CIF.
    """
    rows, receipts = [], []
    if not manifest:
        raise ValueError("No result records")
    ids = set()
    required = {"peptide_chain_plddt", "global_iptm"}
    if required - set(mapping):
        raise ValueError("Metric mapping must include peptide pLDDT and global ipTM")
    for item in manifest:
        if item["task_name"] in ids:
            raise ValueError("Duplicate task_name")
        ids.add(item["task_name"])
        metrics = read_json(item["metrics_json"])
        interface = read_json(item["interface_json"])
        if sha256(item["cif_file"]) != interface["structure_sha256"]:
            raise ValueError("Interface record belongs to a different CIF")
        row = {"task_name": item["task_name"], "peptide_helm": item["peptide_helm"],
               "cif_file": item["cif_file"], "target_interface_residues": ",".join(interface["target_interface_residues"]),
               "interface_threshold_angstrom": interface["interface_threshold_angstrom"],
               "interface_numbering": interface["numbering"], "target_chain": interface["target_chain"]}
        for column, path in mapping.items():
            row[column] = finite(lookup(metrics, path), column)
        finite(row["peptide_chain_plddt"], "peptide pLDDT", 0, 100)
        finite(row["global_iptm"], "ipTM", 0, 1)
        rows.append(row)
        receipts.append({"task_name":item["task_name"],"metrics_sha256":sha256(item["metrics_json"]),
                         "interface_sha256":sha256(item["interface_json"]),"cif_sha256":interface["structure_sha256"]})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output/"metrics.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output/"provenance.json", {"mapping":mapping,"records":receipts})
    return {"collected":len(rows)}
