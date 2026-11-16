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


def _rank_deficient(H):
    """Mask of matrices whose correlation-scaled form has an eigenvalue below a rank tolerance.

    Scaling by the diagonal makes the test independent of feature units; the tolerance is
    100 * k * eps of the dtype (float64: ~1e-13 * k, float32: ~1e-5 * k).
    """
    d = torch.diagonal(H, dim1=-2, dim2=-1)
    dinv = torch.where(d > 0, d.clamp(min=torch.finfo(H.dtype).tiny).rsqrt(), torch.zeros_like(d))
    scaled = H * dinv.unsqueeze(-1) * dinv.unsqueeze(-2)
    tol = 100 * H.shape[-1] * torch.finfo(H.dtype).eps
    return ~(torch.linalg.eigvalsh(scaled)[..., 0] > tol)


def spd_inverse(H, singular=None):
    """Batched inverse and a mask of numerically singular matrices, whose inverse is NaN.

    A singular information matrix has no finite covariance, so its standard errors are NaN.
    singular optionally marks matrices already known to be singular.
    """
    chol, info = torch.linalg.cholesky_ex(H)
    bad = (info != 0) | _rank_deficient(H)
    if singular is not None:
        bad = bad | singular
    eye = torch.eye(H.shape[-1], device=H.device, dtype=H.dtype)
    inv = torch.cholesky_inverse(torch.where(bad[:, None, None], eye, chol))  # raises on zero diag
    return torch.where(bad[:, None, None], torch.full_like(inv, float("nan")), inv), bad
