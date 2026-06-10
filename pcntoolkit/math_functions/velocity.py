"""Algorithms for longitudinal (velocity-centile) modelling.

This modules has the mathematical implementation of the correlation matrix, 
thrivelines and conditional forecasting.
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


def compute_correlation_matrix(
    data: NormData,
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
    # Flatten the xarray dataset into one table per observation.
    df = data.to_dataframe()[
        ["X", "Z", "batch_effects", "subject_ids"]
    ].droplevel(level=0, axis=1)

    # Stop early when there are no repeated subjects at all.
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
    grps: dict[int, list[int]] = defaultdict(list)
    # Fill that mapping using the grouped dataframe indices.
    grps.update(df.groupby(covariate_name).indices)

    # Read the largest age present in the longitudinal cohort.
    max_age = int(max(grps.keys()))
    # Read response-variable names so matrices stay labeled.
    response_vars = data.response_vars.to_numpy()
    # Count how many response variables must be processed.
    n_responsevars = len(response_vars)

    # Start from the identity (a subject is perfectly correlated with itself).
    cors = np.tile(np.eye(max_age + 1), (n_responsevars, 1, 1))

    # Visit all age pairs that fall within the requested bandwidth.
    for age1, age2 in offset_indices(max_age, bandwidth):
        # Match the same subjects observed at both ages.
        merged = pd.merge(
            df.iloc[grps[age1]],
            df.iloc[grps[age2]],
            how="inner",
            on="subject_ids",
        )
        # Only estimate a correlation when enough paired subjects exist.
        if len(merged) >= 4:
            # Compute one correlation per response variable.
            for i, rv in enumerate(response_vars):
                cors[i, age2, age1] = cors[i, age1, age2] = (
                    merged[f"{rv}_x"].corr(merged[f"{rv}_y"])
                )
        # Mark sparse age pairs as missing so regression can fill them.
        elif age1 != age2:
            cors[:, age2, age1] = cors[:, age1, age2] = np.nan

    # Fill sparse off-diagonal entries using the regression model.
    newcors = fill_missing(bandwidth, cors)
    # fill_missing only populates off-diagonal (offset >= 1) entries, so
    # restore the diagonal: a subject's z-score is perfectly correlated
    # with itself.
    # Reset each diagonal entry to a perfect self-correlation.
    for rv in range(n_responsevars):
        np.fill_diagonal(newcors[rv], 1.0)
    # Return the matrix as a labeled xarray object.
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

# ------------------------------------------------------------------- #
# Internals for compute_correlation_matrix
# ------------------------------------------------------------------- #


def fill_missing(bandwidth: int, cors: np.ndarray) -> np.ndarray:
    """Fill in missing correlation values by regressing on age.

    Empirical correlations can only be estimated for age pairs that have enough repeated
    measurements, sparsely sampled age pairs (within a chosen bandwidth) are filled
    in by regressing the Fisher-transformed correlations on age following
    Buuren (2023).

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
    # Work on the Fisher scale where correlations behave more linearly.
    f_cors = fisher_transform(cors)
    # Read the largest age represented in the matrix.
    max_age = f_cors.shape[1] - 1
    # Prepare an output array on the Fisher scale.
    newcors = np.zeros_like(f_cors)
    # Fit one regression model per response variable.
    for rv in range(f_cors.shape[0]):
        # Build the regression table for this response variable.
        Phi = design_matrix(bandwidth, f_cors[rv])
        # Keep only rows where an empirical correlation is available.
        Xy = Phi.dropna(axis=0, inplace=False)
        # Fit Buuren's regression without an extra intercept.
        regmodel = LinearRegression(fit_intercept=False).fit(
            Xy.drop(columns="y", inplace=False), y=Xy[["y"]]
        )
        # Predict Fisher correlations for every allowed age pair.
        y_pred = regmodel.predict(Phi.drop(columns="y"))
        # Write the predictions symmetrically into the matrix.
        for i, (age1, age2) in enumerate(offset_indices(max_age, bandwidth)):
            newcors[rv, age1, age2] = newcors[rv, age2, age1] = (
                y_pred[i].item()
            )
    # Inverse Fisher transform (tanh)
    # Convert predicted Fisher values back to ordinary correlations.
    return np.tanh(newcors)


def fisher_transform(cor: np.ndarray) -> np.ndarray:
    """Fisher z-transform."""
    # Keep correlations away from exactly +/-1 before transforming.
    epsilon = 1e-13
    # Clip values into the open interval required by the transform.
    cor = np.clip(cor, -1 + epsilon, 1 - epsilon)
    # Convert correlations into Fisher z values.
    return 0.5 * np.log((1 + cor) / (1 - cor))


def offset_indices(max_age: int, bandwidth: int):
    """Yield upper-triangular (age1, age2) pairs within ``bandwidth``.

    E.g. ``offset_indices(3, 2)`` yields (0,1), (0,2), (1,2), (1,3), (2,3).
    """
    # Start from an empty age-by-age mask.
    acc = np.zeros((max_age + 1, max_age + 1))
    # Mark all off-diagonal upper-triangle pairs as candidates.
    acc[np.triu_indices(max_age + 1, 1)] = 1
    # Remove pairs whose age gap is larger than the bandwidth.
    acc[np.triu_indices(max_age + 1, bandwidth + 1)] = 0
    # Yield the remaining valid age pairs one by one.
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
    # Read the largest age represented in the matrix.
    max_age = Sigma.shape[0] - 1
    # Build a base age axis used for all offsets.
    ages = np.arange(max_age)
    # Collect one small table per age offset.
    dfs = []
    # Create the regression rows for each age gap separately.
    for offset in range(1, bandwidth + 1):
        # Keep only starting ages that fit this age gap.
        ages_i = ages[: max_age - offset + 1]
        # Start a design table indexed by the first age.
        df_i = pd.DataFrame(index=ages_i)
        # Add the intercept term from Buuren's formulation.
        df_i["v0"] = 1
        # Add the mean-age term for this pair.
        df_i["V1"] = np.log(ages_i + (offset / 2))
        # Add the age-gap term.
        df_i["V2"] = np.log(offset)
        # Add the inverse-gap term.
        df_i["V3"] = 1 / offset
        # Add the interaction term between age and age gap.
        df_i["V4"] = df_i["V1"] * df_i["V2"]
        # Add the nonlinear age term.
        df_i["V5"] = df_i["V1"] ** 2
        # Store the observed Fisher correlation as the target.
        df_i["y"] = np.diagonal(Sigma, offset)
        # Save this offset-specific block.
        dfs.append(df_i)
    # Stack all offset blocks into one regression table.
    return pd.concat(dfs, axis=0)
