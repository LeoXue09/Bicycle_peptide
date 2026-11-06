"""Batch covalent inputs and portable plans for a declared prediction backend."""
from pathlib import Path
import re
from .hf3 import build_task
from .io import sha256, write_json

SCHEMA = "bicycle-helm-covalent-v1"


def build_batch(records, target, linker, output, head_to_tail=False, allow_extra_cys=False):
    """Turn accepted screening records into covalent tasks with traceable inputs."""
    if not records:
        raise ValueError("No screened candidates")
    prepared = []
    names = set()
    for record in records:
        if record["screen"]["accepted"] is not True:
            raise ValueError("Task generation requires accepted screening records")
        name = record["task_name"]
        if name in names:
            raise ValueError("Duplicate candidate task name")
        names.add(name)
        if sha256(record["pdb_path"]) != record["structure_sha256"]:
            raise ValueError("Candidate structure changed after screening")
        task = build_task(name, record["sequence"], target["sequence"], linker,
                          [p + 1 for p in record["screen"]["positions_0based"]],
                          head_to_tail=head_to_tail, allow_extra_cys=allow_extra_cys)
        prepared.append((record, task))
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for record, task in prepared:
        path = output / (task["job_name"] + ".json")
        write_json(path, task)
        jobs.append({"task_name": task["job_name"], "input_json": str(path),
                     "input_sha256": sha256(path), "peptide_sequence": record["sequence"],
                     "source_structure_sha256": record["structure_sha256"]})
    manifest = {"schema": SCHEMA, "jobs": jobs, "target": target,
                "linker": linker, "head_to_tail": head_to_tail}
    write_json(output / "manifest.json", manifest)
    return manifest


def plan_batch(manifest, backend, output):
    """Build argv without executing it or silently changing covalent inputs."""
    if manifest.get("schema") != SCHEMA or backend.get("input_schema") != SCHEMA:
        raise ValueError(f"Backend and task manifest must declare {SCHEMA}")
    command = backend["argv"]
    if not isinstance(command, list) or not command or not all(isinstance(a, str) for a in command):
        raise ValueError("Backend argv must be a nonempty list of strings")
    text = " ".join(command)
    if "{input_json}" not in text or "{output_dir}" not in text:
        raise ValueError("Backend command requires input_json and output_dir placeholders")
    if not isinstance(backend.get("gpu_required"), bool):
        raise ValueError("Backend must explicitly declare gpu_required")
    expected = backend["expected_files"]
    if not isinstance(expected, list) or not expected:
        raise ValueError("Backend must declare expected output files")
    for name in expected:
        if not isinstance(name,str) or not name or Path(name).is_absolute() or ".." in Path(name).parts or name == ".":
            raise ValueError("Expected outputs must be relative paths within the job directory")
    output = Path(output).absolute()
    plans = []
    names = set()
    if not manifest["jobs"]:
        raise ValueError("Empty task manifest")
    for job in manifest["jobs"]:
        name = job["task_name"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", ".."} or name in names:
            raise ValueError("Unsafe or duplicate task name")
        names.add(name)
        source = Path(job["input_json"]).absolute()
        if sha256(source) != job["input_sha256"]:
            raise ValueError("Prediction input changed after task construction")
        destination = output / "outputs" / name
        argv = [a.replace("{input_json}", str(source)).replace("{output_dir}", str(destination)) for a in command]
        plans.append({"stage": "covalent_prediction", "task_name": name, "input_schema": SCHEMA,
                      "argv": argv, "cwd": str(Path(backend["cwd"]).absolute()),
                      "gpu_required": backend["gpu_required"], "env": backend.get("env", {}),
                      "inputs": [str(source), *backend.get("input_files", [])],
                      "engine_roots": backend.get("engine_roots", []),
                      "expected_outputs": [str(destination / f) for f in expected]})
    output.mkdir(parents=True, exist_ok=False)
    paths = []
    for plan in plans:
        path = output / (plan["task_name"] + ".plan.json")
        write_json(path, plan)
        paths.append(str(path))
    write_json(output / "plans.json", {"schema": SCHEMA, "plans": paths})
    return {"plans": paths}
