"""Algorithms for longitudinal (velocity-centile) modelling.

This modules has the mathematical implementation of the correlation matrix, 
thrivelines and conditional forecasting.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
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


# ------------------------------------------------------------------- #
# Thrivelines
# ------------------------------------------------------------------- #


def get_thrive_lines(
    correlations: xr.DataArray,
    start_z: xr.DataArray | float,
    z_thrive: float = -1.96,
) -> xr.DataArray:
    """Thriveline propagator from Bayer et al. (2026).

        Z_{next} = Z_{current} * r + sqrt(1 - r^2) * z_thrive

    Parameters
    ----------
    correlations : xr.DataArray
        Pearson correlations for each step, earliest timepoint first.
        Must have dimension ``hop``.
    start_z : xr.DataArray or float
        Z-score at the anchor timepoint.
    z_thrive : float, default -1.96
        Thrive shrinkage term added at each step.

    Returns
    -------
    xr.DataArray
        Z-scores along the segment, dimension ``offset``.
    """
    if "hop" not in correlations.dims:
        raise ValueError("correlations must have dimension 'hop'.")

    z0 = float(start_z.item() if isinstance(start_z, xr.DataArray) else start_z)
    z_path = [z0]
    current = z0
    for r in correlations.transpose("hop").values:
        current = current * float(r) + math.sqrt(1.0 - float(r) ** 2) * z_thrive
        z_path.append(current)

    timepoint_diff = correlations.attrs.get("timepoint_diff", 1)
    return xr.DataArray(
        z_path,
        dims=("offset",),
        coords={"offset": np.arange(len(z_path)) * timepoint_diff},
        attrs={"z_thrive": z_thrive, "timepoint_diff": timepoint_diff},
    )


def compute_thrivelines(
    R: xr.DataArray,
    *,
    data: NormData | None = None,
    covariate: str = "age",
    timepoint_diff: int | float = 1,
    z_thrive: float = -1.96,
    propagate: Callable[[xr.DataArray, xr.DataArray | float, float], xr.DataArray] = get_thrive_lines,
    anchor_step: int | float = 1,
    z_anchor_start: int | float = -3,
    z_anchor_end: int | float = 4,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Propagate Z-scores along thriveline segments.

    ``R`` is typically from :func:`compute_correlation_matrix` or
    :meth:`~pcntoolkit.longitudinal_score.zgain_score.ZGainScore.get_correlation_matrix`.
    Thrivelines are computed for every label in ``R.response_vars``.

    Parameters
    ----------
    R : xr.DataArray
        Z-score correlation matrix
        ``(response_vars, f"{covariate}_1", f"{covariate}_2")``.
    data : NormData, optional
        Observation anchors (requires predicted ``Z``). When omitted, a
        covariate × Z grid is built for overlay plots.
    covariate : str, default "age"
        Covariate used to read anchor ages from ``data.X``.
    timepoint_diff : int or float, default 1
        Covariate step between timepoints on each segment.
    z_thrive : float, default -1.96
        Passed through to ``propagate``.
    propagate : callable, default :func:`get_thrive_lines`
        ``propagate(hop_correlations, start_z, z_thrive) -> xr.DataArray``.
    anchor_step, z_anchor_start, z_anchor_end
        Grid-mode parameters, used only when ``data`` is omitted.

    Returns
    -------
    thrive_Z, thrive_X : xr.DataArray
        Dims ``(segment, response_vars, offset)``.
    """
    if timepoint_diff <= 0:
        raise ValueError("timepoint_diff must be > 0.")

    R, response_vars = _response_vars_from_R(R)
    age_dim_later, age_dim_earlier = _covariate_age_dims(R.isel(response_vars=0, drop=True))
    max_covariate = float(R.coords[age_dim_later].max())
    end_covariate = max_covariate + timepoint_diff

    if data is not None:
        start_ages, start_z = _anchors_from_normdata(data, covariate, response_vars)
    else:
        start_ages, start_z = _grid_anchors(
            0, end_covariate, anchor_step, z_anchor_start, z_anchor_end
        )

    start_ages, start_z = _drop_out_of_range_anchors(
        start_ages, start_z, max_covariate, timepoint_diff
    )

    n_segments = start_ages.sizes["segment"]
    offsets = xr.DataArray([0, timepoint_diff], dims="offset")
    thrive_Z = xr.DataArray(
        np.full((n_segments, len(response_vars), 2), np.nan),
        dims=("segment", "response_vars", "offset"),
        coords={
            "segment": start_ages.coords["segment"],
            "response_vars": response_vars,
            "offset": offsets,
        },
        attrs={
            "propagate": getattr(propagate, "__name__", str(propagate)),
            "timepoint_diff": timepoint_diff,
        },
    )
    thrive_X = xr.DataArray(
        np.full((n_segments, len(response_vars), 2), np.nan),
        dims=("segment", "response_vars", "offset"),
        coords={
            "segment": start_ages.coords["segment"],
            "response_vars": response_vars,
            "offset": offsets,
        },
    )

    for rv_idx, rv in enumerate(response_vars):
        R_rv = R.sel(response_vars=rv, drop=True)
        for seg_idx in range(n_segments):
            covariate_anchor = start_ages.isel(segment=seg_idx).item()
            covariate_to = covariate_anchor + timepoint_diff
            if "response_vars" in start_z.dims:
                z_anchor = start_z.isel(segment=seg_idx, response_vars=rv_idx)
            else:
                z_anchor = start_z.isel(segment=seg_idx)
            r = _lookup_correlation(
                R_rv, covariate_anchor, covariate_to, age_dim_later, age_dim_earlier
            )
            if r is None:
                continue
            hop_rs = xr.DataArray(
                [r],
                dims=("hop",),
                coords={"hop": [0]},
                attrs={"timepoint_diff": timepoint_diff},
            )
            z_path = propagate(hop_rs, z_anchor, z_thrive)
            if "offset" not in z_path.dims:
                raise ValueError(
                    "propagate must return an xr.DataArray with dimension 'offset'."
                )
            thrive_Z[{"segment": seg_idx, "response_vars": rv_idx}] = z_path.values
            thrive_X[{"segment": seg_idx, "response_vars": rv_idx}] = (
                covariate_anchor + offsets
            ).values

    thrive_Z = thrive_Z.assign_coords(start_age=("segment", start_ages.values))
    thrive_X = thrive_X.assign_coords(start_age=("segment", start_ages.values))
    if "response_vars" in start_z.dims:
        thrive_Z = thrive_Z.assign_coords(start_z=start_z)
        thrive_X = thrive_X.assign_coords(start_z=start_z)
    else:
        thrive_Z = thrive_Z.assign_coords(start_z=("segment", start_z.values))
        thrive_X = thrive_X.assign_coords(start_z=("segment", start_z.values))
    return thrive_Z, thrive_X


def _covariate_age_dims(R: xr.DataArray) -> tuple[str, str]:
    age_dims = [d for d in R.dims if d.endswith("_1") or d.endswith("_2")]
    if len(age_dims) != 2:
        raise ValueError(
            f"Expected two age dimensions ending in '_1' and '_2', got {R.dims}."
        )
    return age_dims[0], age_dims[1]


def _lookup_correlation(
    R_rv: xr.DataArray,
    covariate_from: int | float,
    covariate_to: int | float,
    age_dim_later: str,
    age_dim_earlier: str,
) -> float | None:
    later_vals = R_rv.coords[age_dim_later].values
    earlier_vals = R_rv.coords[age_dim_earlier].values
    if covariate_to not in later_vals or covariate_from not in earlier_vals:
        return None
    return float(
        R_rv.sel(
            {age_dim_later: covariate_to, age_dim_earlier: covariate_from},
            drop=True,
        ).item()
    )


def _anchors_from_normdata(
    data: NormData,
    covariate: str,
    response_vars: list[str],
) -> tuple[xr.DataArray, xr.DataArray]:
    if "Z" not in data:
        raise ValueError("NormData must contain predicted Z-scores before computing thrivelines.")

    covariates = [str(c) for c in data.covariates.values]
    if covariate not in covariates:
        raise ValueError(f"covariate '{covariate}' not found in NormData covariates: {covariates}.")

    data_response_vars = [str(r) for r in data.response_vars.values]
    missing = [rv for rv in response_vars if rv not in data_response_vars]
    if missing:
        raise ValueError(
            f"response variables {missing} not found in NormData: {data_response_vars}."
        )

    ages = np.round(data.X.sel(covariates=covariate).values.astype(float))
    zs = data.Z.sel(response_vars=response_vars).values.astype(float)
    if zs.ndim == 1:
        zs = zs[:, np.newaxis]
    observations = data.observations.values

    return (
        xr.DataArray(ages, dims=("segment",), coords={"segment": observations}, name="start_age"),
        xr.DataArray(
            zs,
            dims=("segment", "response_vars"),
            coords={"segment": observations, "response_vars": response_vars},
            name="start_z",
        ),
    )


def _grid_anchors(
    start_covariate: int | float,
    end_covariate: int | float,
    anchor_step: int | float,
    z_anchor_start: int | float,
    z_anchor_end: int | float,
) -> tuple[xr.DataArray, xr.DataArray]:
    covariate_anchors = np.arange(start_covariate, end_covariate, anchor_step)
    z_anchors = np.arange(z_anchor_start, z_anchor_end, dtype=float)
    age_grid, z_grid = np.meshgrid(covariate_anchors, z_anchors, indexing="ij")
    segment = np.arange(age_grid.size)
    return (
        xr.DataArray(age_grid.ravel(), dims=("segment",), coords={"segment": segment}, name="start_age"),
        xr.DataArray(z_grid.ravel(), dims=("segment",), coords={"segment": segment}, name="start_z"),
    )


def _response_vars_from_R(R: xr.DataArray) -> tuple[xr.DataArray, list[str]]:
    if "response_vars" not in R.dims:
        if "response_vars" in R.coords and R.coords["response_vars"].ndim == 0:
            label = str(R.coords["response_vars"].item())
            R = R.expand_dims(response_vars=[label])
        elif R.name is not None:
            R = R.expand_dims(response_vars=[str(R.name)])
        elif "response_var" in R.attrs:
            R = R.expand_dims(response_vars=[str(R.attrs["response_var"])])
        else:
            raise ValueError(
                "R has no response_vars dimension. Pass the matrix from "
                "compute_correlation_matrix, or select one region with "
                "R.sel(response_vars=name) so the label is kept as a coordinate."
            )
    return R, [str(r) for r in R.response_vars.values]


def _drop_out_of_range_anchors(
    start_ages: xr.DataArray,
    start_z: xr.DataArray,
    max_covariate: int | float,
    timepoint_diff: int | float,
) -> tuple[xr.DataArray, xr.DataArray]:
    valid = start_ages.values + timepoint_diff <= max_covariate
    return start_ages.isel(segment=valid), start_z.isel(segment=valid)
