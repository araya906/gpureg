"""Batched linear algebra for symmetric positive (semi)definite matrices."""
import torch


def spd_solve(H, g):
    """Solve H x = g per batch; pseudo-inverse fallback where Cholesky fails."""
    chol, info = torch.linalg.cholesky_ex(H)
    x = torch.cholesky_solve(g.unsqueeze(-1), chol).squeeze(-1)
    bad = info != 0
    if bool(bad.any()):
        x[bad] = (torch.linalg.pinv(H[bad], hermitian=True) @ g[bad].unsqueeze(-1)).squeeze(-1)
    return x


def spd_inverse(H):
    """Batched inverse and a mask of matrices that were numerically singular."""
    chol, info = torch.linalg.cholesky_ex(H)
    inv = torch.cholesky_inverse(chol)
    bad = info != 0
    if bool(bad.any()):
        inv[bad] = torch.linalg.pinv(H[bad], hermitian=True)
    return inv, bad
