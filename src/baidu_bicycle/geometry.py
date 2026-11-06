"""CA triangle screening. Indices are zero-based; distances are Angstroms.

CA geometry is a screening proxy, not proof of covalent chemical feasibility.
"""
from itertools import combinations
from math import dist, isfinite


def triangle_search(coords, mode="flexible", positions=None, minimum=6.0,
                    maximum=10.0, tolerance=0.0, spacing=3, min_length=13):
    if mode not in {"flexible", "fixed_ends"}:
        raise ValueError("Unknown triangle search mode")
    if not all(isfinite(x) for x in (minimum, maximum, tolerance)) or not 0 <= tolerance < minimum < maximum:
        raise ValueError("Invalid distance interval/tolerance")
    if not isinstance(spacing, int) or spacing < 1 or min_length < 3:
        raise ValueError("Invalid spacing/minimum length")
    if any(len(c) != 3 or not all(isfinite(v) for v in c) for c in coords):
        raise ValueError("Coordinates must be finite 3-vectors")
    n = len(coords)
    if n < min_length:
        return {"accepted": False, "reason": "peptide_too_short", "length": n}
    pool = list(range(n)) if positions is None else sorted(set(positions))
    if any(not isinstance(p, int) or not 0 <= p < n for p in pool):
        raise ValueError("Triangle positions outside peptide")
    candidates = combinations(pool, 3)
    best = None
    closest = None
    midpoint = (minimum + maximum) / 2
    for triplet in candidates:
        a, b, c = triplet
        if b - a < spacing or c - b < spacing:
            continue
        if mode == "fixed_ends" and (a != 0 or c != n - 1):
            continue
        distances = [dist(coords[i], coords[j]) for i, j in ((a,b),(a,c),(b,c))]
        score = sum((d - midpoint) ** 2 for d in distances)
        record = {"positions_0based": list(triplet), "distances_angstrom": distances,
                  "distance_score": score}
        if closest is None or score < closest[0]:
            closest = (score, record)
        if all(minimum - tolerance < d < maximum + tolerance for d in distances):
            # Break ties by loop balance, then residue indices.
            key = (score, abs((b-a)-(c-b)), triplet)
            if best is None or key < best[0]:
                best = (key, record)
    if best:
        return {"accepted": True, "length": n, **best[1]}
    return {"accepted": False, "reason": "no_valid_triangle", "length": n,
            "closest": closest[1] if closest else None}


def screen_sequence(seq, coords, allowed_cys_counts=(3,), **kwargs):
    from .io import sequence
    sequence(seq)
    if len(seq) != len(coords):
        raise ValueError("Sequence/coordinate length mismatch")
    if seq.count("C") not in allowed_cys_counts:
        return {"accepted": False, "reason": "cysteine_count", "count": seq.count("C")}
    return triangle_search(coords, positions=[i for i, aa in enumerate(seq) if aa == "C"], **kwargs)
