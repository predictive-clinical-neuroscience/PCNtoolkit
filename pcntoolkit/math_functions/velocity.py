"""Z-score correlation matrix for longitudinal (velocity-centile) modelling.

This module estimates, per response variable, the correlation between a
subject's z-scores at two different ages. These correlations describe how
strongly an individual's normative position is expected to persist over time,
and form the basis of the z-gain (generalised velocity centile) score
(Bayer et al., 2026).

The correlation matrix is indexed by integer ages. Because empirical
correlations can only be estimated for age pairs that have enough repeated
measurements, sparsely sampled age pairs (within a chosen bandwidth) are filled
in by regressing the Fisher-transformed correlations on age following
Buuren (2023).
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LinearRegression

if TYPE_CHECKING:
    from pcntoolkit.dataio.norm_data import NormData


def fisher_transform(cor: np.ndarray) -> np.ndarray:
    """Fisher z-transform; clips to (-1, 1) for numerical safety."""
    epsilon = 1e-13
    cor = np.clip(cor, -1 + epsilon, 1 - epsilon)
    return 0.5 * np.log((1 + cor) / (1 - cor))


def offset_indices(max_age: int, bandwidth: int):
    """Yield upper-triangular (age1, age2) pairs within ``bandwidth``.

    E.g. ``offset_indices(3, 2)`` yields (0,1), (0,2), (1,2), (1,3), (2,3).
    """
    acc = np.zeros((max_age + 1, max_age + 1))
    acc[np.triu_indices(max_age + 1, 1)] = 1
    acc[np.triu_indices(max_age + 1, bandwidth + 1)] = 0
    for pair in zip(*np.where(acc)):
        yield pair


def design_matrix(bandwidth: int, Sigma: np.ndarray) -> pd.DataFrame:
    """Construct the regression design matrix to fill missing correlations.

    Follows Buuren, S. "Evaluation and prediction of individual growth
    trajectories." Ann. Hum. Biol. 50, 247-257 (2023).

    Parameters
    ----------
    bandwidth : int
        The bandwidth (max age offset) for which correlations were computed.
    Sigma : np.ndarray
        Correlation matrix (possibly with missing values) for a single
        response variable. The 0th column represents an age of 0.

    Returns
    -------
    pd.DataFrame
        Design matrix with regressors and a (possibly NaN) target column ``y``.
    """
    max_age = Sigma.shape[0] - 1
    ages = np.arange(max_age)
    dfs = []
    for offset in range(1, bandwidth + 1):
        ages_i = ages[: max_age - offset + 1]
        df_i = pd.DataFrame(index=ages_i)
        df_i["v0"] = 1
        df_i["V1"] = np.log(ages_i + (offset / 2))
        df_i["V2"] = np.log(offset)
        df_i["V3"] = 1 / offset
        df_i["V4"] = df_i["V1"] * df_i["V2"]
        df_i["V5"] = df_i["V1"] ** 2
        df_i["y"] = np.diagonal(Sigma, offset)
        dfs.append(df_i)
    return pd.concat(dfs, axis=0)


def fill_missing(bandwidth: int, cors: np.ndarray) -> np.ndarray:
    """Fill in missing correlation values by regressing on age.

    Parameters
    ----------
    bandwidth : int
        The bandwidth within which the indices are filled in.
    cors : np.ndarray
        Possibly incomplete correlation matrix of shape
        ``[n_response_vars, n_ages, n_ages]``.

    Returns
    -------
    np.ndarray
        Matrix completed with predicted values.
    """
    f_cors = fisher_transform(cors)
    max_age = f_cors.shape[1] - 1
    newcors = np.zeros_like(f_cors)
    for rv in range(f_cors.shape[0]):
        Phi = design_matrix(bandwidth, f_cors[rv])
        Xy = Phi.dropna(axis=0, inplace=False)
        regmodel = LinearRegression(fit_intercept=False).fit(
            Xy.drop(columns="y", inplace=False), y=Xy[["y"]]
        )
        y_pred = regmodel.predict(Phi.drop(columns="y"))
        for i, (age1, age2) in enumerate(offset_indices(max_age, bandwidth)):
            newcors[rv, age1, age2] = newcors[rv, age2, age1] = (
                y_pred[i].item()
            )
    # Inverse Fisher transform (tanh)
    return np.tanh(newcors)


def get_correlation_matrix(
    data: "NormData",
    bandwidth: int,
    covariate_name: str = "age",
) -> xr.DataArray:
    """Compute z-score correlations between visits of the same subject by age.

    The covariate (typically age) is binned to integer years before computing
    correlations, so the resulting matrix is indexed by integer ages.

    Parameters
    ----------
    data : NormData
        Longitudinal data containing covariates (X), predicted z-scores (Z),
        batch effects and subject ids.
    bandwidth : int
        The age-offset range within which correlations are computed; larger
        offsets are interpolated by :func:`fill_missing`.
    covariate_name : str, default "age"
        Covariate used to index the correlation matrix.

    Returns
    -------
    xr.DataArray
        Correlations of shape ``[n_response_vars, n_ages, n_ages]`` with dims
        ``(response_vars, f"{covariate_name}_1", f"{covariate_name}_2")``.
    """
    df = data.to_dataframe()[
        ["X", "Z", "batch_effects", "subject_ids"]
    ].droplevel(level=0, axis=1)

    if not df["subject_ids"].duplicated().any():
        raise ValueError(
            "Cannot compute correlation matrix: the dataset is "
            "cross-sectional. A z-score correlation matrix requires "
            "longitudinal data (multiple observations per subject_id "
            "at different ages)."
        )

    # Bin the covariate to integer years so the matrix can be indexed by age.
    df[covariate_name] = np.round(
        df[covariate_name].astype(float)
    ).astype(int)

    # Map each integer age to the row indices of observations at that age.
    grps: dict = defaultdict(list)
    grps.update(df.groupby(covariate_name).indices)

    max_age = int(max(grps.keys()))
    response_vars = data.response_vars.to_numpy()
    n_responsevars = len(response_vars)

    # Start from the identity (a subject is perfectly correlated with itself).
    cors = np.tile(np.eye(max_age + 1), (n_responsevars, 1, 1))

    for age1, age2 in offset_indices(max_age, bandwidth):
        merged = pd.merge(
            df.iloc[grps[age1]],
            df.iloc[grps[age2]],
            how="inner",
            on="subject_ids",
        )
        if len(merged) >= 4:
            for i, rv in enumerate(response_vars):
                cors[i, age2, age1] = cors[i, age1, age2] = (
                    merged[f"{rv}_x"].corr(merged[f"{rv}_y"])
                )
        elif age1 != age2:
            cors[:, age2, age1] = cors[:, age1, age2] = np.nan

    newcors = fill_missing(bandwidth, cors)
    # fill_missing only populates off-diagonal (offset >= 1) entries, so
    # restore the diagonal: a subject's z-score is perfectly correlated
    # with itself.
    for rv in range(n_responsevars):
        np.fill_diagonal(newcors[rv], 1.0)
    return xr.DataArray(
        newcors,
        dims=(
            "response_vars",
            f"{covariate_name}_1",
            f"{covariate_name}_2",
        ),
        coords={
            "response_vars": response_vars,
            f"{covariate_name}_1": np.arange(cors.shape[1]),
            f"{covariate_name}_2": np.arange(cors.shape[1]),
        },
    )
