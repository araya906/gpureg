"""Coercion and validation of user inputs (numpy arrays or pandas objects)."""
import numpy as np
import pandas as pd


def as_matrix(obj, prefix):
    """Return (2-D float array, column names) from a DataFrame, Series or array."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        return obj.to_numpy(dtype=float), [str(c) for c in obj.columns]
    arr = np.asarray(obj, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr, [f"{prefix}{j}" for j in range(arr.shape[1])]


def as_vector(obj, name):
    """Return a 1-D float array, rejecting missing values."""
    arr = np.asarray(obj, dtype=float).ravel()
    if np.isnan(arr).any():
        raise ValueError(f"{name} contains missing values; remove those rows first.")
    return arr


def check_rows(arr, name, n):
    """Check that a matrix has n rows."""
    if arr.shape[0] != n:
        raise ValueError(f"{name} has {arr.shape[0]} rows, expected {n}.")


def require_complete(arr, name, n):
    """Check row count and absence of missing values."""
    check_rows(arr, name, n)
    if np.isnan(arr).any():
        raise ValueError(f"{name} contains missing values; remove those rows first.")


def covariate_matrix(covariates, n):
    """Complete (n, p) covariate matrix, or an (n, 0) matrix when there are none."""
    if covariates is None:
        return np.empty((n, 0))
    arr, _ = as_matrix(covariates, "cov")
    require_complete(arr, "covariates", n)
    return arr
