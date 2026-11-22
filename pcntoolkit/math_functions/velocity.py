"""Algorithms for longitudinal (velocity-centile) modelling.

This modules has the mathematical implementation of the correlation matrix,
thrivelines and conditional forecasting.
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

if TYPE_CHECKING:
    from pcntoolkit.dataio.norm_data import NormData
    from pcntoolkit.normative_model import NormativeModel

# Same default centiles as NormativeModel.compute_centiles, so thrivelines
# start on the centile curves that are plotted by default.
DEFAULT_THRIVELINE_CENTILES = [0.05, 0.25, 0.5, 0.75, 0.95]

# ------------------------------------------------------------------- #
# Correlation matrix
# ------------------------------------------------------------------- #


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
        Widest age gap, in years, to estimate correlations for.
        Correlations for wider gaps are set to 0.
    covariate_name : str, default "age"
        Covariate used to index the correlation matrix.

    Returns
    -------
    xr.DataArray
        Correlations of shape ``[n_response_vars, n_ages, n_ages]`` with dims
        ``(response_vars, f"{covariate_name}_1", f"{covariate_name}_2")``.

    Raises
    ------
    ValueError
        If the data is cross-sectional, or if no pair of ages within
        ``bandwidth`` has at least 4 subjects scanned at both ages.
    """
    # Flatten the xarray dataset into one table per observation.
    df = data.to_dataframe()[["X", "Z", "batch_effects", "subject_ids"]].droplevel(
        level=0, axis=1
    )

    # Stop early when there are no repeated subjects at all.
    if not df["subject_ids"].duplicated().any():
        raise ValueError(
            "Cannot compute correlation matrix: the dataset is "
            "cross-sectional. A z-score correlation matrix requires "
            "longitudinal data (multiple observations per subject_id "
            "at different ages)."
        )

    # Bin the covariate to integer years so the matrix can be indexed by age.
    df[covariate_name] = np.round(df[covariate_name].astype(float)).astype(int)

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

    # Count the age pairs with enough subjects to estimate a correlation.
    n_estimated = 0
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
            n_estimated += 1
            # Compute one correlation per response variable.
            for i, rv in enumerate(response_vars):
                cors[i, age2, age1] = cors[i, age1, age2] = merged[f"{rv}_x"].corr(
                    merged[f"{rv}_y"]
                )
        # Mark sparse age pairs as missing so regression can fill them.
        elif age1 != age2:
            cors[:, age2, age1] = cors[:, age1, age2] = np.nan

    # Without a single estimated correlation, the regression below has
    # nothing to learn from.
    if n_estimated == 0:
        raise ValueError(
            "Cannot compute correlation matrix: no pair of ages has at least "
            "4 subjects scanned at both ages. Use more longitudinal subjects, "
            "or load a precomputed matrix with CorrelationMatrix.load()."
        )

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
        # List the age pairs in the same order as the rows of Phi:
        # first all pairs 1 year apart, then all pairs 2 years apart, etc.
        pairs = [
            (age1, age1 + offset)
            for offset in range(1, bandwidth + 1)
            for age1 in range(max_age - offset + 1)
        ]
        # Put each predicted correlation in the cell of its age pair.
        for (age1, age2), y in zip(pairs, y_pred.ravel(), strict=True):
            newcors[rv, age1, age2] = newcors[rv, age2, age1] = y
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


def thriveline_end_z(
    r: float,
    start_z: float,
    zgain_threshold: float = -1.96,
) -> float:
    """Compute the z-score where one thriveline ends (Bayer et al., 2026).

    The line ends at the later z-score whose z-gain is exactly
    ``zgain_threshold``. Solving ``z_gain = (z_end − r·z_start) / sqrt(1 − r²)``
    for ``z_end`` gives

        z_end = r · z_start + sqrt(1 − r²) · zgain_threshold

    Parameters
    ----------
    r : float
        Correlation of z-scores between the start and end age of the line,
        between -1 and 1.
    start_z : float
        z-score where the line starts.
    zgain_threshold : float, default -1.96
        z-gain the line follows.

    Returns
    -------
    float
        z-score where the line ends.
    """
    return r * start_z + math.sqrt(1.0 - r**2) * zgain_threshold


def compute_thriveline_z(
    R: xr.DataArray,
    *,
    zgain_threshold: float = -1.96,
    thriveline_span: int = 1,
    thriveline_spacing: int = 1,
    centiles: list[float] | np.ndarray | None = None,
    covariate_range: tuple[int, int] | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute thrivelines, in z-scores, from a correlation matrix.

    Parameters
    ----------
    R : xr.DataArray
        Correlation matrix with dimensions
        ``(response_vars, <covariate>_1, <covariate>_2)``. Timepoints with a zero or
        missing correlation get no thriveline, and a warning is raised.
    zgain_threshold : float, default -1.96
        z-gain each line follows.
    thriveline_span : int, default 1
        How far each thriveline stretches along the x-axis.
    thriveline_spacing : int, default 1
        Years between neighbouring thrivelines along the x-axis.
    centiles : array-like of float, optional
        Centile curves the lines start on, between 0 and 1. Defaults to
        ``[0.05, 0.25, 0.5, 0.75, 0.95]``.
    covariate_range : tuple of int, optional
        ``(min, max)`` ages to draw lines for. Defaults to every age in ``R``.

    Returns
    -------
    thrive_Z, thrive_X : xr.DataArray
        Z-scores and ages of each line's start and end point, with dimensions
        ``(segment, response_vars, offset)``. Lines that could not be computed
        are NaN.

    Raises
    ------
    ValueError
        If ``thriveline_span`` is not positive, or if a centile is not
        strictly between 0 and 1.
    """
    # Reject non-positive spans because a line must move forward in time.
    if thriveline_span <= 0:
        raise ValueError("thriveline_span must be > 0.")

    if centiles is None:
        centiles = DEFAULT_THRIVELINE_CENTILES
    centiles = np.asarray(centiles, dtype=float)
    # The 0th and 100th centiles map to infinite z-scores.
    if np.any((centiles <= 0) | (centiles >= 1)):
        raise ValueError(
            f"centiles must be strictly between 0 and 1, got {centiles.tolist()}."
        )

    # Ensure R carries a response_vars dimension and collect its labels.
    R, response_vars = _response_vars_from_R(R)
    # Optionally restrict R and the anchor grid to a covariate sub-range.
    if covariate_range is not None:
        R = _slice_r_to_covariate_range(R, covariate_range)
        min_covariate, max_covariate = covariate_range
    else:
        # Otherwise use the full covariate span represented in R.
        min_covariate, max_covariate = _covariate_bounds_from_R(R)

    # Identify which matrix axes correspond to later and earlier covariate values.
    age_dim_later, age_dim_earlier = _covariate_age_dims(
        R.isel(response_vars=0, drop=True)
    )
    # Extend the grid upper bound so the last anchor can still take one forward step.
    end_covariate = max_covariate + thriveline_span

    # Build the Cartesian grid of (covariate anchor, starting z-score) pairs.
    start_ages, start_z = _grid_anchors(
        min_covariate,
        end_covariate,
        thriveline_spacing,
        norm.ppf(centiles),
    )

    # Drop anchors whose forward step would fall outside the correlation matrix.
    start_ages, start_z = _drop_out_of_range_anchors(
        start_ages, start_z, min_covariate, max_covariate, thriveline_span
    )

    # Count how many thriveline segments remain after filtering.
    n_segments = start_ages.sizes["segment"]
    # Each segment has exactly two offsets: the anchor and one step forward.
    offsets = xr.DataArray([0, thriveline_span], dims="offset")
    # Preallocate z-score output; NaN marks segments with missing correlations.
    thrive_Z = xr.DataArray(
        np.full((n_segments, len(response_vars), 2), np.nan),
        dims=("segment", "response_vars", "offset"),
        coords={
            "segment": start_ages.coords["segment"],
            "response_vars": response_vars,
            "offset": offsets,
        },
        attrs={
            "zgain_threshold": zgain_threshold,
            "thriveline_span": thriveline_span,
        },
    )
    # Preallocate covariate output on the same grid as thrive_Z.
    thrive_X = xr.DataArray(
        np.full((n_segments, len(response_vars), 2), np.nan),
        dims=("segment", "response_vars", "offset"),
        coords={
            "segment": start_ages.coords["segment"],
            "response_vars": response_vars,
            "offset": offsets,
        },
    )

    # Compute every segment independently for each response variable.
    skipped_pairs: set[tuple[int, int]] = set()
    for rv_idx, rv in enumerate(response_vars):
        # Select the 2-D correlation slice for this response variable.
        R_rv = R.sel(response_vars=rv, drop=True)
        for seg_idx in range(n_segments):
            # Read the covariate value where this segment starts.
            covariate_anchor = start_ages.isel(segment=seg_idx).item()
            # The forward timepoint is one fixed covariate step later.
            covariate_to = covariate_anchor + thriveline_span
            # Read the starting z-score for this segment (shared across regions).
            z_anchor = float(start_z.isel(segment=seg_idx).item())
            # Look up the Pearson r between anchor and forward covariate values.
            r = _lookup_correlation(
                R_rv, covariate_anchor, covariate_to, age_dim_later, age_dim_earlier
            )
            # Skip segments whose correlation is missing or zero (uncomputed offset).
            if r is None or r == 0.0:
                skipped_pairs.add((int(covariate_anchor), int(covariate_to)))
                continue
            # Find where the line ends, thriveline_span years later.
            z_end = thriveline_end_z(r, z_anchor, zgain_threshold)
            # Store the start and end z-scores for this segment and response variable.
            thrive_Z[{"segment": seg_idx, "response_vars": rv_idx}] = [z_anchor, z_end]
            # Store the matching covariate coordinates (anchor and forward point).
            thrive_X[{"segment": seg_idx, "response_vars": rv_idx}] = (
                covariate_anchor + offsets
            ).values

    # Name the skipped age pairs, since their lines are left out of the output.
    if skipped_pairs:
        pairs = ", ".join(f"{a}->{b}" for a, b in sorted(skipped_pairs))
        warnings.warn(
            "Thriveline computation cannot be completed for all requested "
            f"lines: the correlation matrix has no correlation for ages {pairs}. "
            "Those lines are left out.",
            UserWarning,
            stacklevel=2,
        )

    # Attach the starting covariate of each segment as a segment coordinate.
    thrive_Z = thrive_Z.assign_coords(start_age=("segment", start_ages.values))
    thrive_X = thrive_X.assign_coords(start_age=("segment", start_ages.values))
    # Attach the starting z-score of each segment as a segment coordinate.
    thrive_Z = thrive_Z.assign_coords(start_z=("segment", start_z.values))
    thrive_X = thrive_X.assign_coords(start_z=("segment", start_z.values))
    return thrive_Z, thrive_X


def compute_thriveline_y(
    model: "NormativeModel",
    thrive_Z: xr.DataArray | xr.Dataset,
    thrive_X: xr.DataArray | None = None,
    *,
    covariate: str | None = None,
    batch_effects: dict[str, str] | None = None,
) -> xr.DataArray:
    """Convert thrivelines from z-scores to the response-variable Y coordinates.

    Uses the normative model to turn each point's (age, z-score) into the
    value on the y-axis of the centile chart, e.g. cortical thickness in mm.

    Parameters
    ----------
    model : NormativeModel
        The fitted normative model.
    thrive_Z : xr.DataArray or xr.Dataset
        Z-scores from ``compute_thriveline_z``, with dimensions
        ``(segment, response_vars, offset)``. Can also be a Dataset holding
        both ``Z`` and ``X``; ``thrive_X`` can then be left out.
    thrive_X : xr.DataArray, optional
        Ages from ``compute_thriveline_z``, same shape as ``thrive_Z``.
    covariate : str, optional
        Name of the covariate on the x-axis, e.g. ``"age"``. Only needed
        when the model has more than one covariate. That way it knows to put 
        all other covariates at the middle of their range.
    batch_effects : dict of str, optional
        Batch effect level to draw the lines for, e.g. ``{"site": "A"}``.
        Defaults to the level that appears first in the model's training
        data. Ignored when the model has no batch effects.

    Returns
    -------
    thrive_Y : xr.DataArray
        Response variable value at each point, same dimensions as ``thrive_Z``. Lines
        that could not be computed stay NaN.
    """
    # Identify the covariate that advances along each thriveline segment.
    covariates = list(model.covariates)
    if len(covariates) == 1:
        thrive_covariate = covariates[0]
    else:
        if covariate is None:
            raise ValueError(
                "covariate must be specified when the model has multiple "
                f"covariates: {covariates}."
            )
        if covariate not in covariates:
            raise ValueError(
                f"covariate '{covariate}' is not among model covariates: {covariates}."
            )
        thrive_covariate = covariate

    thrive_Z, thrive_X = _resolve_thriveline_inputs(thrive_Z, thrive_X)
    # Any covariate that does not change along the segment is held fixed.
    other_covariates = [c for c in covariates if c != thrive_covariate]
    fixed_covariates = _resolve_fixed_covariates(model, other_covariates)

    response_vars = [str(r) for r in thrive_Z.coords["response_vars"].values]
    # Preallocate Y on the same (segment, response_vars, offset) grid as Z.
    thrive_Y = xr.DataArray(
        np.full(thrive_Z.shape, np.nan),
        dims=thrive_Z.dims,
        coords=thrive_Z.coords,
        attrs=thrive_Z.attrs,
        name="thrive_Y",
    )

    n_segments = thrive_Z.sizes["segment"]
    # One reference batch label per segment for the backward pass.
    be_encoded = _encode_batch_effects(model, batch_effects or {}, n_segments)

    for rv in response_vars:
        z_rv = thrive_Z.sel(response_vars=rv, drop=True)
        x_rv = thrive_X.sel(response_vars=rv, drop=True)
        y_rv = np.full((n_segments, z_rv.sizes["offset"]), np.nan)

        # Map each offset slice (anchor and forward point) to response scale.
        for o_idx, off in enumerate(z_rv.coords["offset"].values):
            ages = x_rv.sel(offset=off).values.astype(float)
            z_vals = z_rv.sel(offset=off).values.astype(float)
            # Segments with missing correlations stay NaN in Z and are skipped.
            valid = np.isfinite(ages) & np.isfinite(z_vals)
            if not valid.any():
                continue

            n_valid = int(valid.sum())
            obs_coord = np.arange(n_valid)
            X_scaled = np.zeros((n_valid, len(covariates)))
            for i, cov in enumerate(covariates):
                if cov == thrive_covariate:
                    col = ages[valid].reshape(-1, 1)
                else:
                    col = np.full((n_valid, 1), fixed_covariates[cov])
                X_scaled[:, i] = model.inscalers[cov].transform(col).ravel()

            X_da = xr.DataArray(
                X_scaled,
                dims=("observations", "covariates"),
                coords={"observations": obs_coord, "covariates": covariates},
            )
            Z_da = xr.DataArray(
                z_vals[valid],
                dims=("observations",),
                coords={"observations": obs_coord},
            )
            be_slice = (
                be_encoded.isel(observations=np.where(valid)[0])
                if be_encoded is not None
                else None
            )
            # Invert the regional normative map: (X, Z) -> scaled Y, then unscale.
            y_scaled = model[rv].backward(X_da, be_slice, Z_da).values
            y_vals = (
                model.outscalers[rv].inverse_transform(y_scaled.reshape(-1, 1)).ravel()
            )
            y_rv[valid, o_idx] = y_vals

        thrive_Y.loc[{"response_vars": rv}] = y_rv

    return thrive_Y


# Public long-form schema: one row per point on a thriveline segment.
THRIVELINE_DF_COLUMNS = (
    "segment",
    "response_var",
    "offset",
    "X",
    "Z",
    "Y",
)


def validate_thrivelines(thrivelines: pd.DataFrame) -> None:
    """Check that a pre-computed thriveline table can be plotted."""
    if not isinstance(thrivelines, pd.DataFrame):
        raise TypeError(
            "thrivelines must be a pandas DataFrame from "
            "VelocityModel.compute_thrivelines()."
        )
    missing = [col for col in THRIVELINE_DF_COLUMNS if col not in thrivelines.columns]
    if missing:
        raise ValueError(
            f"thrivelines is missing columns {missing}. "
            "Compute it with VelocityModel.compute_thrivelines() first."
        )


def thrivelines_to_dataframe(
    thrive_Z: xr.DataArray,
    thrive_X: xr.DataArray,
    thrive_Y: xr.DataArray,
) -> pd.DataFrame:
    """Convert propagated thriveline arrays to a long-form table.

    Each row is one point on one segment for one response variable. Columns are
    ``segment``, ``response_var``, ``offset``, ``X``, ``Z``, and ``Y``.
    Segments that could not be computed (NaN) are left out.

    Parameters
    ----------
    thrive_Z, thrive_X, thrive_Y : xr.DataArray
        Arrays with dimensions ``(segment, response_vars, offset)`` as returned
        by ``compute_thriveline_z`` and ``compute_thriveline_y``.

    Returns
    -------
    pd.DataFrame
        Long-form thriveline table suitable for inspection, filtering, and
        plotting.
    """
    rows: list[dict[str, object]] = []
    offsets = thrive_Z.coords["offset"].values

    for rv in thrive_Z.coords["response_vars"].values:
        z_rv = thrive_Z.sel(response_vars=rv, drop=True)
        x_rv = thrive_X.sel(response_vars=rv, drop=True)
        y_rv = thrive_Y.sel(response_vars=rv, drop=True)
        for seg_idx in range(z_rv.sizes["segment"]):
            # Skip segments that could not be computed; the warning names them.
            if np.isnan(z_rv.isel(segment=seg_idx).values).all():
                continue
            segment = int(z_rv.coords["segment"].values[seg_idx])
            # Each segment has two points: offset 0 (anchor) and offset thriveline_span.
            for off_idx, offset in enumerate(offsets):
                rows.append(
                    {
                        "segment": segment,
                        "response_var": str(rv),
                        "offset": int(offset),
                        "X": float(x_rv.isel(segment=seg_idx, offset=off_idx).item()),
                        "Z": float(z_rv.isel(segment=seg_idx, offset=off_idx).item()),
                        "Y": float(y_rv.isel(segment=seg_idx, offset=off_idx).item()),
                    }
                )

    df = pd.DataFrame(rows, columns=list(THRIVELINE_DF_COLUMNS))
    # Keep the settings the lines were computed with for later inspection.
    for key in ("thriveline_span", "zgain_threshold"):
        if key in thrive_Z.attrs:
            df.attrs[key] = thrive_Z.attrs[key]
    return df


def _resolve_thriveline_inputs(
    thrive_Z: xr.DataArray | xr.Dataset,
    thrive_X: xr.DataArray | None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return ``(thrive_Z, thrive_X)`` from separate arrays or a Dataset."""
    # Accept either paired DataArrays or a bundled Dataset with Z and X variables.
    if isinstance(thrive_Z, xr.Dataset):
        if "Z" not in thrive_Z or "X" not in thrive_Z:
            raise ValueError(
                "When thrive_Z is an xr.Dataset it must contain data variables "
                "'Z' and 'X'."
            )
        return thrive_Z["Z"], thrive_Z["X"]
    if thrive_X is None:
        raise ValueError(
            "thrive_X is required when thrive_Z is not a Dataset with 'Z' and 'X'."
        )
    return thrive_Z, thrive_X


def _resolve_reference_batch_effects(
    model: "NormativeModel",
    batch_effects: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return reference batch labels used for thriveline backward passes.

    Explicit entries in ``batch_effects`` override the default. Otherwise each
    dimension uses the first allowed level in ``model.unique_batch_effects``
    (same rule as ``plot_thrivelines``).
    """
    if not model.unique_batch_effects:
        return {}
    overrides = batch_effects or {}
    # Default to the first allowed level so thrivelines align with plot_thrivelines.
    return {
        k: overrides[k] if k in overrides else model.unique_batch_effects[k][0]
        for k in sorted(model.unique_batch_effects.keys())
    }


def _encode_batch_effects(
    model: "NormativeModel",
    batch_effects: dict[str, str],
    n_obs: int,
) -> xr.DataArray | None:
    """Build a model-encoded batch-effect array for thriveline backward passes.

    Each thriveline segment is evaluated at the reference batch effect resolved
    by ``_resolve_reference_batch_effects``.
    """
    if not model.unique_batch_effects:
        return None
    be_vals = _resolve_reference_batch_effects(model, batch_effects)
    be_keys = sorted(be_vals.keys())
    obs_coord = np.arange(n_obs)
    # Repeat the same reference batch labels for every thriveline segment.
    raw_be = xr.DataArray(
        np.tile([[str(be_vals[k]) for k in be_keys]], (n_obs, 1)),
        dims=("observations", "batch_effect_dims"),
        coords={"observations": obs_coord, "batch_effect_dims": be_keys},
    )
    return model.map_batch_effects(raw_be)


def _covariate_bounds_from_R(R: xr.DataArray) -> tuple[int, int]:
    """Return the minimum and maximum integer covariate values present in ``R``."""
    # Resolve the two covariate axis names on a single-region slice of R.
    age_dim_later, age_dim_earlier = _covariate_age_dims(
        R.isel(response_vars=0, drop=True)
    )
    # Collect every covariate coordinate value from both matrix axes.
    ages = np.concatenate(
        [
            R.coords[age_dim_later].values.ravel(),
            R.coords[age_dim_earlier].values.ravel(),
        ]
    )
    # Return the span as integer bounds for grid placement and slicing.
    return int(np.min(ages)), int(np.max(ages))


def _slice_r_to_covariate_range(
    R: xr.DataArray,
    covariate_range: tuple[int, int],
) -> xr.DataArray:
    """Restrict ``R`` to covariate values inside ``[lo, hi]`` on both axes."""
    # Unpack the inclusive lower and upper covariate bounds.
    lo, hi = covariate_range
    # Resolve which dimensions index later and earlier covariate values.
    age_dim_later, age_dim_earlier = _covariate_age_dims(
        R.isel(response_vars=0, drop=True)
    )
    # Slice both covariate axes symmetrically to the requested range.
    return R.sel({age_dim_later: slice(lo, hi), age_dim_earlier: slice(lo, hi)})


def _covariate_age_dims(R: xr.DataArray) -> tuple[str, str]:
    """Return the later and earlier covariate dimension names in ``R``."""
    # Correlation matrices name the two covariate axes with _1 and _2 suffixes.
    age_dims = [d for d in R.dims if d.endswith("_1") or d.endswith("_2")]
    # Require exactly two such axes so lookup is unambiguous.
    if len(age_dims) != 2:
        raise ValueError(
            f"Expected two age dimensions ending in '_1' and '_2', got {R.dims}."
        )
    # Return the pair in matrix order (later axis first, earlier second).
    return age_dims[0], age_dims[1]


def _lookup_correlation(
    R_rv: xr.DataArray,
    covariate_from: int,
    covariate_to: int,
    age_dim_later: str,
    age_dim_earlier: str,
) -> float | None:
    """Look up Pearson r between ``covariate_from`` and ``covariate_to`` in ``R_rv``."""
    # Read the covariate coordinates available on the later-time axis.
    later_vals = R_rv.coords[age_dim_later].values
    # Read the covariate coordinates available on the earlier-time axis.
    earlier_vals = R_rv.coords[age_dim_earlier].values
    # Return None when either endpoint is absent from the matrix coordinates.
    if covariate_to not in later_vals or covariate_from not in earlier_vals:
        return None
    # Select the matrix cell and return the correlation as a float.
    return float(
        R_rv.sel(
            {age_dim_later: covariate_to, age_dim_earlier: covariate_from},
            drop=True,
        ).item()
    )


def _grid_anchors(
    start_covariate: int,
    end_covariate: int,
    thriveline_spacing: int,
    z_values: np.ndarray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Build a Cartesian grid of covariate and z-score anchor points."""
    # Place covariate anchors from start up to (but not including) end_covariate.
    covariate_anchors = np.arange(start_covariate, end_covariate, thriveline_spacing)
    # Form every (covariate, z-score) pair on the overlay grid.
    age_grid, z_grid = np.meshgrid(covariate_anchors, z_values, indexing="ij")
    # Assign a flat segment index to each grid cell.
    segment = np.arange(age_grid.size)
    # Return covariate and z anchors as parallel segment-indexed arrays.
    return (
        xr.DataArray(
            age_grid.ravel(),
            dims=("segment",),
            coords={"segment": segment},
            name="start_age",
        ),
        xr.DataArray(
            z_grid.ravel(),
            dims=("segment",),
            coords={"segment": segment},
            name="start_z",
        ),
    )


def _response_vars_from_R(R: xr.DataArray) -> tuple[xr.DataArray, list[str]]:
    """Ensure ``R`` has a ``response_vars`` dimension and return its labels."""
    # Most callers pass a matrix that already includes response_vars.
    if "response_vars" not in R.dims:
        # Recover a single-region label from a scalar response_vars coordinate.
        if "response_vars" in R.coords and R.coords["response_vars"].ndim == 0:
            label = str(R.coords["response_vars"].item())
            R = R.expand_dims(response_vars=[label])
        # Fall back to the DataArray name when present.
        elif R.name is not None:
            R = R.expand_dims(response_vars=[str(R.name)])
        # Fall back to a stored attribute label.
        elif "response_var" in R.attrs:
            R = R.expand_dims(response_vars=[str(R.attrs["response_var"])])
        else:
            raise ValueError(
                "R has no response_vars dimension. Pass the matrix from "
                "compute_correlation_matrix, or select one region with "
                "R.sel(response_vars=name) so the label is kept as a coordinate."
            )
    # Return the possibly expanded matrix and string labels for each region.
    return R, [str(r) for r in R.response_vars.values]


def _drop_out_of_range_anchors(
    start_ages: xr.DataArray,
    start_z: xr.DataArray,
    min_covariate: int,
    max_covariate: int,
    thriveline_span: int,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Keep only anchors whose forward step stays inside the correlation matrix."""
    # A segment is valid when the anchor lies in range and the forward step fits.
    valid = (start_ages.values >= min_covariate) & (
        start_ages.values + thriveline_span <= max_covariate
    )
    # Return filtered covariate and z anchors with matching segment indices.
    return start_ages.isel(segment=valid), start_z.isel(segment=valid)


def _resolve_fixed_covariates(
    model: "NormativeModel",
    other_covariates: list[str],
) -> dict[str, float]:
    """
    Set any other covariates to the midpoint of their range in the model.

    Parameters
    ----------
    model : NormativeModel
        Fitted model, read for its covariate ranges.
    other_covariates : list of str
        Covariates that do not advance along the thriveline.

    Returns
    -------
    dict of float
        One value per covariate in ``other_covariates``.

    Raises
    ------
    ValueError
        If the model has no recorded range for one of the covariates.
    """
    ranges = getattr(model, "covariate_ranges", None)
    resolved: dict[str, float] = {}
    for cov in other_covariates:
        if not ranges or cov not in ranges:
            # Throw error for models with no recorded covariate range
            raise ValueError(
                f"No value available for covariate '{cov}': the model has no "
                "recorded range for it. Refit the model, or use one saved with "
                "a newer version of PCNtoolkit."
            )
        resolved[cov] = (ranges[cov]["min"] + ranges[cov]["max"]) / 2
    return resolved
