"""Engine plans, ProteinMPNN preparation and recorded subprocess execution."""
from pathlib import Path
import json
import os
import subprocess
import time
from .io import read_json, write_json, sha256, integer, finite
from .geometry import triangle_search
from .structure import pdb_residues


def rfdiffusion_plan(config, profile, output):
    engine = Path(profile["rfdiffusion_root"]).resolve()
    python = str(Path(profile["rfdiffusion_python"]).absolute())
    length = integer(config["peptide_length"], "peptide length", 3)
    positions = config["triangle_positions_0based"]
    if len(positions) != 3 or len(set(positions)) != 3:
        raise ValueError("Three triangle positions required")
    if any(integer(p, "triangle position") >= length for p in positions):
        raise ValueError("Triangle positions outside peptide")
    scale = finite(config["guide_scale"], "guide scale", 0)
    strength = finite(config["k_dist"], "k_dist", 0)
    if config["potential_version"] != "bicycle_triangle_v1":
        raise ValueError("Choose a supported versioned potential")
    target = Path(config["target_pdb"]).resolve()
    checkpoint = Path(profile["rfdiffusion_checkpoint"]).resolve()
    count = integer(config["num_designs"], "num designs", 1)
    start = integer(config.get("design_startnum", 0), "design start")
    contig = f'{length}-{length} {config["target_contig"]}/0'
    potential = "type:bicycle_triangle_v1," + ",".join(f"pos{i+1}:{p}" for i,p in enumerate(positions)) + f",k_dist:{strength}"
    argv = [python, "-m", "baidu_bicycle.rfdiffusion_entry", "--engine", str(engine),
            f"inference.input_pdb={target}", f"inference.output_prefix={Path(output).resolve() / 'design'}",
            f"inference.ckpt_override_path={checkpoint}",
            f"inference.num_designs={count}", f"inference.design_startnum={start}",
            "inference.deterministic=True", "inference.cyclic=True", "inference.cyc_chains=a",
            f"contigmap.contigs=[{contig}]", "ppi.hotspot_res=["+",".join(config["hotspots"])+"]",
            f"potentials.guiding_potentials=[{potential!r}]", f"potentials.guide_scale={scale}",
            f"potentials.guide_decay={config.get('guide_decay', 'quadratic')}",
            f"diffuser.T={integer(config.get('diffusion_steps',50), 'diffusion steps',1)}",
            f"hydra.run.dir={Path(output).resolve() / 'hydra'}"]
    return {"stage": "rfdiffusion", "argv": argv, "cwd": str(engine),
            "inputs": [str(target), str(checkpoint)], "engine_roots": [str(engine)],
            "expected_outputs": [str(Path(output).resolve()/f"design_{i}.{suffix}")
                                 for i in range(start,start+count) for suffix in ("pdb","trb")],
            "gpu_required": True, "scientific_config": config,
            "env": {"PYTHONPATH": str(Path(__file__).resolve().parents[1])}}


def prepare_mpnn(source, chain, output, profile, mode="flexible", min_length=13,
                 n_sequences=10, temperature=.1, seed=1):
    residues = pdb_residues(source, chain)
    if any(r["icode"] or r["number"] != residues[0]["number"]+i for i,r in enumerate(residues)):
        raise ValueError("MPNN peptide must have consecutive residue IDs without insertion codes")
    if any(set(("N","CA","C","O"))-set(r["atoms"]) for r in residues):
        raise ValueError("ProteinMPNN requires N, CA, C, O for each peptide residue")
    result = triangle_search([r["atoms"]["CA"] for r in residues], mode=mode, min_length=min_length)
    if not result["accepted"]:
        raise ValueError(f"Backbone rejected: {result}")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    selected = {residues[i]["number"] for i in result["positions_0based"]}
    lines = []
    for line in Path(source).read_text().splitlines():
        if line.startswith("ATOM  ") and line[21:22] == chain and int(line[22:26]) in selected:
            # Only backbone survives mutation; an old side chain must not masquerade as CYS.
            if line[12:16].strip() not in {"N","CA","C","O"}:
                continue
            line = line[:17]+"CYS"+line[20:]
        lines.append(line)
    pdb = output / "backbone.pdb"
    pdb.write_text("\n".join(lines)+"\n")
    chains = sorted({line[21:22] for line in lines if line.startswith("ATOM  ")})
    fixed = {c: [] for c in chains}
    fixed[chain] = [p+1 for p in result["positions_0based"]]
    fixed_path = output / "fixed_positions.jsonl"
    # Upstream reads each line independently with json.loads, not a pretty JSON document.
    with fixed_path.open("x") as stream:
        stream.write(json.dumps({pdb.stem: fixed}) + "\n")
    root = Path(profile["proteinmpnn_root"]).resolve()
    weights = Path(profile["proteinmpnn_weights"]).resolve()
    model_name = profile.get("proteinmpnn_model", "v_48_020")
    plan = {"stage": "proteinmpnn", "cwd": str(root), "gpu_required": True,
            "argv": [profile["proteinmpnn_python"], str(root/"protein_mpnn_run.py"),
                     "--pdb_path", str(pdb), "--pdb_path_chains", chain,
                     "--fixed_positions_jsonl", str(fixed_path), "--omit_AAs", "CX",
                     "--num_seq_per_target", str(integer(n_sequences,"sequences",1)),
                     "--sampling_temp", str(finite(temperature,"temperature",1e-9)),
                     "--seed", str(integer(seed,"seed",1)), "--out_folder", str(output/"sequences"),
                     "--path_to_model_weights", str(weights), "--model_name", model_name],
            "inputs": [str(pdb), str(fixed_path), str(weights/f"{model_name}.pt")],
            "engine_roots": [str(root)], "expected_outputs": [str(output/"sequences/seqs/backbone.fa")],
            "source_sha256": sha256(source), "triangle": result}
    write_json(output/"plan.json", plan)
    return plan


def execute(plan, output, gpu=None):
    """No shell; logs, input hashes, engine revisions, failure state, elapsed time.

    The caller chooses a new run directory. No automatic retries or overwrites.
    GPU occupancy is recorded and active processes on the chosen GPU cause refusal.
    """
    if not isinstance(plan["argv"], list) or not all(isinstance(v,str) for v in plan["argv"]):
        raise ValueError("argv must be a list of strings")
    inputs = {p: sha256(p) for p in plan.get("inputs", [])}
    for p in plan.get("expected_outputs", []):
        if Path(p).exists():
            raise FileExistsError(f"Refusing existing output: {p}")
    env = os.environ.copy()
    env.update(plan.get("env", {}))
    occupancy = None
    if plan.get("gpu_required"):
        if gpu is None:
            raise ValueError("Explicit physical GPU index or UUID required")
        selected = subprocess.check_output(["nvidia-smi", "-i", str(gpu),
                    "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
        occupancy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                                            "--format=csv,noheader"], text=True)
        if any(line.split(",")[0].strip() == selected for line in occupancy.splitlines()):
            raise RuntimeError("Selected GPU has active compute processes; choose an available device")
        env["CUDA_VISIBLE_DEVICES"] = selected
    revisions = {}
    for root in plan.get("engine_roots", []):
        revisions[root] = {"commit": subprocess.check_output(["git","-C",root,"rev-parse","HEAD"],text=True).strip(),
                           "status": subprocess.check_output(["git","-C",root,"status","--porcelain"],text=True)}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    record = {"plan": plan, "input_sha256": inputs, "engine_revisions": revisions,
              "gpu": gpu, "occupancy_before": occupancy, "started_unix": started,
              "status": "running"}
    write_json(output/"started.json", record)
    try:
        with (output/"stdout.log").open("w") as out, (output/"stderr.log").open("w") as err:
            result = subprocess.run(plan["argv"], cwd=plan["cwd"], env=env, stdout=out, stderr=err, check=False)
        record["returncode"] = result.returncode
        if result.returncode:
            raise RuntimeError(f"Engine failed with status {result.returncode}; inspect {output}")
        for p in plan.get("expected_outputs", []):
            if not Path(p).is_file() or Path(p).stat().st_size == 0:
                raise RuntimeError(f"Missing/empty expected output: {p}")
        record["output_sha256"] = {p: sha256(p) for p in plan.get("expected_outputs", [])}
        record["status"] = "complete"
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = repr(error)
        raise
    finally:
        record["elapsed_seconds"] = time.time()-started
        write_json(output/"finished.json", record)
    return record
