"""Fit containers and conversion to pandas tables."""
import math
import warnings
from typing import NamedTuple

import pandas as pd
import torch


class BatchFit(NamedTuple):
    beta: torch.Tensor       # (b, k)
    se: torch.Tensor         # (b, k)
    loglik: torch.Tensor     # (b,)
    converged: torch.Tensor  # (b,) bool
    n_iter: torch.Tensor     # (b,)
    singular: torch.Tensor   # (b,) bool, information matrix not invertible
    n_obs: torch.Tensor      # (b,) rows used
    infinite: torch.Tensor   # (b, k) bool, estimate diverges (separation, monotone likelihood)


def wald(beta, se):
    """Wald z statistics and two-sided normal p-values, in float64 on the CPU.

    float32 p would underflow to 0 below ~1e-38, so p is always computed in float64.
    """
    z = beta.detach().cpu().double() / se.detach().cpu().double()
    return z, torch.special.erfc(z.abs() / math.sqrt(2.0))


def _to_numpy(t):
    return t.detach().cpu().numpy()


def term_summary(fit, j):
    """Per-model statistics for coefficient j, as numpy arrays."""
    z, p = wald(fit.beta[:, j], fit.se[:, j])
    return {
        "beta": _to_numpy(fit.beta[:, j]), "se": _to_numpy(fit.se[:, j]),
        "z": _to_numpy(z), "p": _to_numpy(p),
        "n_obs": _to_numpy(fit.n_obs).round().astype(int), "loglik": _to_numpy(fit.loglik),
        "converged": _to_numpy(fit.converged), "n_iter": _to_numpy(fit.n_iter),
        "singular": _to_numpy(fit.singular), "infinite": _to_numpy(fit.infinite.any(1)),
    }


def _warn_infinite(n_flagged, what):
    if n_flagged:
        warnings.warn(f"{n_flagged} {what} may be infinite (separation or monotone likelihood); "
                      "see the 'infinite' column.", stacklevel=3)


def feature_frame(names, rows):
    """One row per screened feature; warns when some models have infinite estimates."""
    _warn_infinite(int(rows["infinite"].sum()), "screened models have a coefficient that")
    return pd.DataFrame({"feature": names, **rows})


def coefficient_frame(names, fit):
    """Coefficient table of a single model; fit diagnostics go in DataFrame.attrs."""
    z, p = wald(fit.beta[0], fit.se[0])
    df = pd.DataFrame({"term": names, "beta": _to_numpy(fit.beta[0]), "se": _to_numpy(fit.se[0]),
                       "z": _to_numpy(z), "p": _to_numpy(p),
                       "infinite": _to_numpy(fit.infinite[0])})
    df.attrs.update(loglik=float(fit.loglik[0]), converged=bool(fit.converged[0]),
                    n_iter=int(fit.n_iter[0]), singular=bool(fit.singular[0]),
                    n_obs=int(round(float(fit.n_obs[0]))))
    _warn_infinite(int(df["infinite"].sum()), "coefficients")
    return df
