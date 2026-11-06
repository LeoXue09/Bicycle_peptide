"""Prepare the downloaded CP-Composer interface, not an invented campaign API."""
from pathlib import Path
import shutil
from .io import write_json, integer, sha256
from .structure import pdb_residues


def prepare(source, receptor_chain, peptide_chain, output, profile, n_samples=5):
    if receptor_chain == peptide_chain:
        raise ValueError("Receptor and peptide chains must differ")
    for chain in (receptor_chain, peptide_chain):
        pdb_residues(source, chain)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/"pdbs").mkdir()
    pdb = output/"pdbs/backbone.pdb"
    shutil.copy2(source, pdb)
    (output/"index.tsv").write_text(f"backbone\t{receptor_chain}\t{peptide_chain}\n")
    config = {"dataset": {"test": {"class": "PromptDataset", "mmap_dir": str(output/"processed"),
              "backbone_only": False, "use_covariance_matrix": True,
              "text_guidance": "The distance between the N-terminal and C-terminal atoms of the peptide is less than 3.80 Å."}},
              "dataloader": {"num_workers": 0, "batch_size": 1}, "backbone_only": False,
              "n_samples": integer(n_samples, "samples", 1), "guidance_strength": 3,
              "CADS_sampler": False, "seq_only": True}
    # JSON is a YAML subset accepted by upstream yaml.safe_load; no YAML dependency here.
    write_json(output/"generation.json", config)
    engine = Path(profile["composer_root"]).resolve()
    python = profile["composer_python"]
    checkpoint = Path(profile["composer_checkpoint"]).resolve()
    preprocess = {"stage": "composer_preprocess", "cwd": str(engine), "gpu_required": False,
                  "argv": [python,"-m","scripts.data_process.process","--index",str(output/"index.tsv"),
                           "--out_dir",str(output/"processed"),"--pocket_th","10.0"],
                  "inputs": [str(pdb),str(output/"index.tsv")], "engine_roots": [str(engine)],
                  "expected_outputs": [str(output/"processed/data.bin"),str(output/"processed/index.txt")]}
    generation = {"stage": "composer_generation", "cwd": str(engine), "gpu_required": True,
                  "argv": [python,str(engine/"generate.py"),"--config",str(output/"generation.json"),
                           "--ckpt",str(checkpoint),"--gpu","0","--n_cpu","1","--save_dir",str(output/"generated")],
                  "env": {"condition": "4"}, "engine_roots": [str(engine)],
                  "inputs": [str(checkpoint),str(output/"generation.json"),str(output/"processed/data.bin"),str(output/"processed/index.txt")],
                  "expected_outputs": [str(output/"generated/results.jsonl")],
                  "qualification": "Source-derived adapter; checkpoint inference not validated"}
    write_json(output/"01_preprocess.plan.json", preprocess)
    write_json(output/"02_generate.plan.json", generation)
    write_json(output/"provenance.json", {"source_sha256": sha256(source),
               "note": "Upstream sequence-only option may still sample coordinates internally; screen emitted candidate structures."})
    return {"preprocess": preprocess, "generation": generation}
