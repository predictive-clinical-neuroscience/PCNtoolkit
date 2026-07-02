"""Algorithms for longitudinal (velocity-centile) modelling.

This modules has the mathematical implementation of the correlation matrix, 
thrivelines and conditional forecasting.
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LinearRegression

if TYPE_CHECKING:
    from pcntoolkit.dataio.norm_data import NormData

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
    """Propagate one thriveline segment using the Bayer et al. (2026) update.

    Each step applies

        Z_{next} = Z_{current} * r + sqrt(1 - r^2) * z_thrive

    where ``r`` is the Pearson correlation between consecutive covariate
    timepoints and ``z_thrive`` pulls the trajectory toward typical growth.

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
    # Require the hop dimension so correlations are ordered in time.
    if "hop" not in correlations.dims:
        raise ValueError("correlations must have dimension 'hop'.")

    # Normalise the starting z-score to a plain Python float.
    z0 = float(start_z.item() if isinstance(start_z, xr.DataArray) else start_z)
    # Store every z-score visited along the segment, beginning at the anchor.
    z_path = [z0]
    # Track the z-score at the current timepoint while propagating forward.
    current = z0
    # Apply the thriveline update once per hop in chronological order.
    for r in correlations.transpose("hop").values:
        # Mix persistence (r) with shrinkage toward z_thrive over one step.
        current = current * float(r) + math.sqrt(1.0 - float(r) ** 2) * z_thrive
        # Append the propagated z-score to the segment path.
        z_path.append(current)

    # Read the covariate spacing between consecutive offsets, defaulting to 1.
    timepoint_diff = correlations.attrs.get("timepoint_diff", 1)
    # Return the full z-path labelled by covariate offset from the anchor.
    return xr.DataArray(
        z_path,
        dims=("offset",),
        coords={"offset": np.arange(len(z_path)) * timepoint_diff},
        attrs={"z_thrive": z_thrive, "timepoint_diff": timepoint_diff},
    )


def compute_thrivelines(
    R: xr.DataArray,
    *,
    timepoint_diff: int = 1,
    z_thrive: float = -1.96,
    propagate: Callable[[xr.DataArray, xr.DataArray | float, float], xr.DataArray] = get_thrive_lines,
    anchor_step: int = 1,
    z_anchor_start: int = -3,
    z_anchor_end: int = 4,
    z_anchors: list[float] | np.ndarray | None = None,
    covariate_range: tuple[int, int] | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Build a grid of thriveline segments from a longitudinal correlation matrix.

    Thrivelines describe how a z-score is expected to change over a fixed
    covariate step (for example, one year), given correlations between
    covariate timepoints. For each label in ``R.response_vars``, the function
    places anchor points on a regular grid of starting covariate values and
    starting z-scores, looks up the correlation between consecutive timepoints
    separated by ``timepoint_diff``, and propagates each anchor forward via
    ``propagate``.

    Parameters
    ----------
    R : xr.DataArray
        Longitudinal correlation matrix between pairs of covariate timepoints,
        with shape ``(response_vars, covariate_1, covariate_2)``. The two
        covariate axes index integer timepoint values (e.g. age in years); each
        entry is the Pearson correlation of z-scores between those timepoints.
        A typical source is :func:`compute_correlation_matrix` or
        :meth:`~pcntoolkit.longitudinal_score.zgain_score.ZGainScore.get_correlation_matrix`.
        To obtain valid thrivelines, the correlations at ``timepoint_diff``
        must be greater than zero; missing or zero entries usually indicate
        uncomputed offsets and those segments are omitted with a warning.
    timepoint_diff : int, default 1
        Covariate step between the two timepoints on each segment (e.g. 1 year).
    z_thrive : float, default -1.96
        Shrinkage term passed through to ``propagate``.
    propagate : callable, default :func:`get_thrive_lines`
        Function that propagates z-scores along one segment:
        ``propagate(hop_correlations, start_z, z_thrive) -> xr.DataArray``.
        The default implements the update of Bayer et al. (2026); alternative
        propagators may be supplied to extend the toolbox.
    anchor_step : int, default 1
        Spacing between starting covariate anchors on the overlay grid.
    z_anchor_start, z_anchor_end : int, default -3 and 4
        Half-open range ``[z_anchor_start, z_anchor_end)`` of integer starting
        z-scores used when ``z_anchors`` is not provided.
    z_anchors : array-like of float, optional
        Explicit starting z-scores (e.g. ``norm.ppf(centiles)``). When given,
        ``z_anchor_start`` and ``z_anchor_end`` are ignored.
    covariate_range : tuple of int, optional
        ``(min, max)`` covariate bounds used to slice ``R`` and place grid
        anchors. When omitted, anchors span the covariate coordinates in ``R``.

    Returns
    -------
    thrive_Z, thrive_X : xr.DataArray
        Propagated z-scores and covariate coordinates, both with dimensions
        ``(segment, response_vars, offset)``. Each segment is one anchor pair
        propagated over ``timepoint_diff``; segments with missing or zero
        correlations are left as NaN and trigger a :class:`UserWarning`.
    """
    # Reject non-positive covariate steps because propagation needs a forward hop.
    if timepoint_diff <= 0:
        raise ValueError("timepoint_diff must be > 0.")

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
    age_dim_later, age_dim_earlier = _covariate_age_dims(R.isel(response_vars=0, drop=True))
    # Extend the grid upper bound so the last anchor can still take one forward step.
    end_covariate = max_covariate + timepoint_diff

    # Build the Cartesian grid of (covariate anchor, starting z-score) pairs.
    start_ages, start_z = _grid_anchors(
        min_covariate,
        end_covariate,
        anchor_step,
        z_anchor_start,
        z_anchor_end,
        z_anchors=z_anchors,
    )

    # Drop anchors whose forward step would fall outside the correlation matrix.
    start_ages, start_z = _drop_out_of_range_anchors(
        start_ages, start_z, min_covariate, max_covariate, timepoint_diff
    )

    # Count how many thriveline segments remain after filtering.
    n_segments = start_ages.sizes["segment"]
    # Each segment has exactly two offsets: the anchor and one step forward.
    offsets = xr.DataArray([0, timepoint_diff], dims="offset")
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
            "propagate": getattr(propagate, "__name__", str(propagate)),
            "timepoint_diff": timepoint_diff,
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

    # Propagate every segment independently for each response variable.
    uncomputed_correlations = False
    for rv_idx, rv in enumerate(response_vars):
        # Select the 2-D correlation slice for this response variable.
        R_rv = R.sel(response_vars=rv, drop=True)
        for seg_idx in range(n_segments):
            # Read the covariate value where this segment starts.
            covariate_anchor = start_ages.isel(segment=seg_idx).item()
            # The forward timepoint is one fixed covariate step later.
            covariate_to = covariate_anchor + timepoint_diff
            # Read the starting z-score for this segment (shared across regions).
            z_anchor = start_z.isel(segment=seg_idx)
            # Look up the Pearson r between anchor and forward covariate values.
            r = _lookup_correlation(
                R_rv, covariate_anchor, covariate_to, age_dim_later, age_dim_earlier
            )
            # Skip segments whose correlation is missing or zero (uncomputed offset).
            if r is None or r == 0.0:
                uncomputed_correlations = True
                continue
            # Package the single-hop correlation for the propagate callable.
            hop_rs = xr.DataArray(
                [r],
                dims=("hop",),
                coords={"hop": [0]},
                attrs={"timepoint_diff": timepoint_diff},
            )
            # Propagate the anchor z-score forward by one covariate step.
            z_path = propagate(hop_rs, z_anchor, z_thrive)
            # Require propagate to return offset-labelled z-scores.
            if "offset" not in z_path.dims:
                raise ValueError(
                    "propagate must return an xr.DataArray with dimension 'offset'."
                )
            # Store the propagated z-scores for this segment and response variable.
            thrive_Z[{"segment": seg_idx, "response_vars": rv_idx}] = z_path.values
            # Store the matching covariate coordinates (anchor and forward point).
            thrive_X[{"segment": seg_idx, "response_vars": rv_idx}] = (
                covariate_anchor + offsets
            ).values

    # Warn when any segment hits an uncomputed age offset in R.
    if uncomputed_correlations:
        warnings.warn(
            "Thriveline computation cannot be completed for all requested "
            f"segments: one or more age pairs at timepoint_diff={timepoint_diff} "
            "have zero or missing correlations in the correlation matrix. This "
            "typically indicates that those age offsets were not estimated "
            "and the thriveline cannot be computed. Affected segments have been omitted.",
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


def _covariate_bounds_from_R(R: xr.DataArray) -> tuple[int, int]:
    """Return the minimum and maximum integer covariate values present in ``R``."""
    # Resolve the two covariate axis names on a single-region slice of R.
    age_dim_later, age_dim_earlier = _covariate_age_dims(R.isel(response_vars=0, drop=True))
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
    age_dim_later, age_dim_earlier = _covariate_age_dims(R.isel(response_vars=0, drop=True))
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
    anchor_step: int,
    z_anchor_start: int,
    z_anchor_end: int,
    z_anchors: list[float] | np.ndarray | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Build a Cartesian grid of covariate and z-score anchor points."""
    # Place covariate anchors from start up to (but not including) end_covariate.
    covariate_anchors = np.arange(start_covariate, end_covariate, anchor_step)
    # Use explicit z anchors when provided, otherwise an integer z-score range.
    if z_anchors is not None:
        z_values = np.asarray(z_anchors, dtype=float)
    else:
        z_values = np.arange(z_anchor_start, z_anchor_end)
    # Form every (covariate, z-score) pair on the overlay grid.
    age_grid, z_grid = np.meshgrid(covariate_anchors, z_values, indexing="ij")
    # Assign a flat segment index to each grid cell.
    segment = np.arange(age_grid.size)
    # Return covariate and z anchors as parallel segment-indexed arrays.
    return (
        xr.DataArray(age_grid.ravel(), dims=("segment",), coords={"segment": segment}, name="start_age"),
        xr.DataArray(z_grid.ravel(), dims=("segment",), coords={"segment": segment}, name="start_z"),
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
    timepoint_diff: int,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Keep only anchors whose forward step stays inside the correlation matrix."""
    # A segment is valid when the anchor lies in range and the forward step fits.
    valid = (start_ages.values >= min_covariate) & (
        start_ages.values + timepoint_diff <= max_covariate
    )
    # Return filtered covariate and z anchors with matching segment indices.
    return start_ages.isel(segment=valid), start_z.isel(segment=valid)
