"""Normalize and screen native ProteinMPNN FASTA or CP-Composer JSONL results."""
import csv
import json
from pathlib import Path
import re
from .io import sha256, write_json, sequence
from .structure import pdb_residues
from .geometry import screen_sequence


def mpnn_records(fasta, pdb, chain, prefix):
    header, parts = None, []
    entries = []
    for line in Path(fasta).read_text().splitlines()+[">"]:
        if line.startswith(">"):
            if header is not None and re.search(r"(?:^|[ ,])sample=",header):
                seq="".join(parts)
                sequence(seq)  # A slash would indicate an unsupported multichain design.
                entries.append({"task_name":f"{prefix}_{len(entries)}", "sequence":seq,
                                "pdb_path":str(Path(pdb).resolve()),"peptide_chain":chain,"engine_header":header})
            header,parts=line[1:],[]
        elif line.strip():
            parts.append(line.strip())
    if not entries:
        raise ValueError("No generated sample= FASTA records; native rows are excluded")
    return entries


def composer_records(jsonl):
    entries=[]
    for line in Path(jsonl).read_text().splitlines():
        if not line.strip(): continue
        record=json.loads(line)
        entries.append({"task_name":f"{record['id']}_{record['number']}","sequence":record['gen_seq'],
                        "pdb_path":record['gen_pdb'],"peptide_chain":record['lig_chain']})
    if not entries: raise ValueError("Empty CP-Composer result file")
    return entries


def screen_records(records, output, min_length=13, allow_four_cys=False):
    rows=[]
    seen=set()
    for record in records:
        if record['task_name'] in seen: raise ValueError("Duplicate task ID")
        seen.add(record['task_name'])
        residues=pdb_residues(record['pdb_path'],record['peptide_chain'])
        coords=[r['atoms']['CA'] for r in residues]
        result=screen_sequence(record['sequence'],coords, min_length=min_length,
                               allowed_cys_counts=(3,4) if allow_four_cys else (3,))
        rows.append({**record,"structure_sha256":sha256(record['pdb_path']),"screen":result})
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'screening.json',rows)
    write_json(output/'accepted.json',[r for r in rows if r['screen']['accepted']])
    return {"total":len(rows),"accepted":sum(r['screen']['accepted'] for r in rows)}
