"""Small, dependency-free records and explicit scientific validation."""
import hashlib
import json
import math
import re
from pathlib import Path

AA = set("ACDEFGHIKLMNPQRSTVWY")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sequence(value):
    if not isinstance(value, str) or not value or set(value) - AA:
        raise ValueError("Expected a nonempty uppercase canonical amino-acid sequence")
    return value


def helm_sequence(helm):
    match = re.search(r"(?:^|\|)PEPTIDE1\{([^}]+)\}", helm)
    if not match:
        raise ValueError("Missing PEPTIDE1 in HELM")
    tokens = match.group(1).split(".")
    if any(len(token) != 1 for token in tokens):
        raise ValueError("Noncanonical HELM monomers need a separate atom mapping")
    return sequence("".join(tokens))


def finite(value, name, low=None, high=None):
    value = float(value)
    if not math.isfinite(value) or (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"Invalid {name}: {value}")
    return value


def integer(value, name, low=0):
    if isinstance(value, bool) or int(value) != value or value < low:
        raise ValueError(f"Invalid {name}: {value}")
    return int(value)
