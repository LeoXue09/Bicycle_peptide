"""Versioned RFdiffusion potential; imported only in the torch runtime.

RFdiffusion ascends potentials, so compute() must return NEGATIVE energy.
The implementation is versioned independently of the FGFR2b reference module.
"""
import math
import torch


class bicycle_triangle_v1:
    def __init__(self, pos1=0, pos2=6, pos3=12, d01=8., d02=8., d12=8.,
                 k_dist=5., binderlen=None):
        values = (pos1, pos2, pos3)
        if any(int(v) != float(v) for v in values):
            raise ValueError("Residue indices must be integers")
        self.positions = tuple(int(v) for v in values)
        if len(set(self.positions)) != 3 or min(self.positions) < 0:
            raise ValueError("Require three distinct nonnegative residue indices")
        self.targets = tuple(float(v) for v in (d01, d02, d12))
        if any(not math.isfinite(v) or v <= 0 for v in self.targets):
            raise ValueError("Target distances must be positive and finite")
        if max(self.targets) >= sum(self.targets) - max(self.targets):
            raise ValueError("Target distances violate the triangle inequality")
        self.k_dist = float(k_dist)
        if not math.isfinite(self.k_dist) or self.k_dist <= 0:
            raise ValueError("k_dist must be finite and positive")
        if binderlen is not None and (int(binderlen) != float(binderlen) or int(binderlen) <= 0):
            raise ValueError("binderlen must be a positive integer")
        self.binderlen = None if binderlen is None else int(binderlen)

    def energy(self, xyz):
        if xyz.ndim != 3 or xyz.shape[1] < 2 or xyz.shape[2] != 3:
            raise ValueError("Expected xyz[L,atoms,3] including CA at atom index 1")
        n = xyz.shape[0] if self.binderlen is None else self.binderlen
        if n > xyz.shape[0] or max(self.positions) >= n:
            raise ValueError("Triangle indices exceed binder length")
        ca = xyz[list(self.positions), 1, :]
        if not torch.isfinite(ca).all():
            raise ValueError("Nonfinite triangle coordinates")
        edges = ca[[0,0,1]] - ca[[1,2,2]]
        distances = (torch.linalg.vector_norm(edges, dim=-1) + 1e-6).clamp(1., 50.)
        error = distances - distances.new_tensor(self.targets)
        absolute = error.abs()
        huber = torch.where(absolute <= 2., 0.5 * error.square(), 2. * (absolute - 1.))
        return self.k_dist * 10. * torch.tanh(huber.sum() / 10.)

    def compute(self, xyz):
        return -self.energy(xyz)


def register():
    """Register in this Python process without editing a shared checkout."""
    from rfdiffusion.potentials import potentials
    name = "bicycle_triangle_v1"
    if name in potentials.implemented_potentials and potentials.implemented_potentials[name] is not bicycle_triangle_v1:
        raise RuntimeError("Potential name already registered by another implementation")
    potentials.implemented_potentials[name] = bicycle_triangle_v1
    potentials.require_binderlen.add(name)
