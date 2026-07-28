from dataclasses import dataclass
from time import perf_counter
import torch
from .operator import apply_negative_laplacian


@dataclass
class PoissonSolveInfo:
    converged: bool
    iterations: int
    initial_residual: float
    final_residual: float
    relative_residual: float
    elapsed_seconds: float


def solve_pcg(b, mask, spacing, initial_guess=None, relative_tolerance=1e-5, absolute_tolerance=0.0, max_iterations=200):
    start = perf_counter()
    x = torch.zeros_like(b) if initial_guess is None else initial_guess.clone() * mask
    apply = lambda value: apply_negative_laplacian(value, mask, spacing)
    r = (b - apply(x)) * mask
    b_norm = torch.linalg.vector_norm(b * mask)
    initial = torch.linalg.vector_norm(r)
    tiny = torch.finfo(b.dtype).tiny
    threshold = max(absolute_tolerance, relative_tolerance * max(float(b_norm), tiny))
    if float(initial) <= threshold:
        return x, PoissonSolveInfo(True, 0, float(initial), float(initial), float(initial/max(b_norm, tiny)), perf_counter()-start)
    diagonal = 2 * sum(1 / d**2 for d in spacing)
    z = r / diagonal
    p = z.clone()
    rz = torch.sum(r * z)
    converged = False
    final = initial
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        ap = apply(p)
        denominator = torch.sum(p * ap)
        if not torch.isfinite(denominator) or abs(float(denominator)) < tiny:
            break
        alpha = rz / denominator
        x += alpha * p
        r -= alpha * ap
        final = torch.linalg.vector_norm(r)
        if float(final) <= threshold:
            converged = True
            break
        z = r / diagonal
        rz_new = torch.sum(r * z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x * mask, PoissonSolveInfo(converged, iterations, float(initial), float(final), float(final/max(b_norm, tiny)), perf_counter()-start)
