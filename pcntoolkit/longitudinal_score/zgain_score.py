from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from pcntoolkit.math_functions.velocity import (
    compute_correlation_matrix,
    compute_thriveline_y,
    compute_thrivelines,
    propagate_thriveline_z,
    thrivelines_to_dataframe,
)

from .longitudinal_score import LongitudinalScore

# Import the heavy classes only during type checking.
if TYPE_CHECKING:
    from pcntoolkit.dataio.norm_data import NormData
    from pcntoolkit.normative_model import NormativeModel


class ZGainScore(LongitudinalScore):
    """Velocity centiles from Bayer et al., 2026 (preprint).

    The z-gain score asks whether a subject's later z-score is surprising once
    their earlier z-score is known. If the typical age-to-age correlation is
    ``r``, the score is

        z_gain = (z_later − r·z_earlier) / sqrt(1 − r²)

    It works with any regression model and any number of visits. When more than
    two visits are available, the score uses the most recent transition (i.e.
    the latest visit conditioned on the immediately preceding visit).
    TODO: Discuss with Johanna if this choice makes sense.

    TODO: Using on the full visit history and not only the most recent
    transition can be done with ``conditional_forecast()``

    Parameters
    ----------
    normative_model : NormativeModel
        A fitted normative model.
    reference_data : NormData
        Longitudinal reference cohort used to learn how z-scores
        correlate across ages.
    subject_id_col : str
        Name of the column that identifies subjects inside both
        ``reference_data`` and the ``score_data``.
    bandwidth : int, default 5
        Age-offset range, in years, for direct correlation estimates.
    covariate : str, default "age"
        The dimension for which the correlations are calculated in the
        correlation matrix.
    max_correlation : float, default 0.99
        Upper bound used to keep the correlation away from 1 and 
        the denominator away from zero. 

    Attributes
    ----------
    correlation_matrix : xr.DataArray | None
        The z-score correlation matrix used by this score. ``None`` until it is
        computed on first use (via :meth:`get_correlation_matrix`, which is also
        called by :meth:`score` and :meth:`get_thrivelines`). Once computed
        it is cached and reused.
    zgain : xr.DataArray | None
        The most recent z-gain scores produced by :meth:`score`. ``None`` until
        :meth:`score` has been called at least once. It stores the same
        ``xr.DataArray`` that :meth:`score` returns, so the result can be
        retrieved later even if the return value was not saved.
    thrivelines : pd.DataFrame | None
        The most recent thrivelines produced by :meth:`get_thrivelines`.
        ``None`` until :meth:`get_thrivelines` has been called at least
        once. A long-form table with columns ``segment``, ``start_age``,
        ``start_z``, ``response_var``, ``offset``, ``X``, ``Z``, and ``Y``.
    """

    def __init__(
        self,
        normative_model: NormativeModel,
        reference_data: NormData,
        subject_id_col: str,
        bandwidth: int = 5,
        covariate: str = "age",
        max_correlation: float = 0.99,
    ):
        # Reuse the shared setup from the base class.
        super().__init__(normative_model, reference_data, subject_id_col)

        # Run the checks
        self._check_is_predicted(self.reference_data)
        self._check_is_longitudinal(self.reference_data)

        self.bandwidth = bandwidth
        self.covariate = covariate
        self.max_correlation = max_correlation

        # Keep the clipping threshold inside a mathematically safe range.
        if not 0.0 < self.max_correlation < 1.0:
            # Explain that the denominator must stay strictly positive.
            raise ValueError(
                "max_correlation must be strictly between 0 and 1."
            )

        # Cache the correlation matrix after first use.
        self.correlation_matrix: xr.DataArray | None = None

        # Hold the most recent z-gain scores; filled in by score().
        self.zgain: xr.DataArray | None = None

        # Hold the most recent thrivelines; filled in by get_thrivelines().
        self.thrivelines: pd.DataFrame | None = None

    def get_correlation_matrix(self) -> xr.DataArray:
        """Estimate and cache the z-score correlation matrix. It computes
        the matrix once and then reuses it for all subsequent calls."""
        if self.correlation_matrix is None:
            self.correlation_matrix = compute_correlation_matrix(
                self.reference_data, self.bandwidth, self.covariate
            )
        return self.correlation_matrix

    def get_thrivelines(
        self,
        *,
        timepoint_diff: int = 1,
        z_thrive: float = -1.96,
        propagate: Callable[
            [xr.DataArray, xr.DataArray | float, float], xr.DataArray
        ] = propagate_thriveline_z,
        anchor_step: int = 1,
        z_anchor_start: int = -3,
        z_anchor_end: int = 4,
        z_anchors: list[float] | np.ndarray | None = None,
        covariate_range: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        """Estimate and cache thrivelines from this score's correlation matrix.

        This wraps :func:`~pcntoolkit.math_functions.velocity.compute_thrivelines`
        and :func:`~pcntoolkit.math_functions.velocity.compute_thriveline_y`. It
        first ensures the z-score correlation matrix is computed and stored,
        propagates thrivelines in Z-space, then maps them to response-scale Y
        via :attr:`normative_model`. Non-thrive covariates are fixed using
        :attr:`reference_data`; batch effects use the first allowed level from
        the normative model (same rule as
        :func:`~pcntoolkit.util.plotter.plot_centiles_advanced`). If the object
        has no correlation matrix yet, one is computed and a warning is issued.

        Parameters
        ----------
        timepoint_diff : int, default 1
            Covariate step between the two timepoints on each thriveline
            segment (e.g. one year).
        z_thrive : float, default -1.96
            Shrinkage term passed to the thriveline propagation update.
        propagate : callable, default :func:`~pcntoolkit.math_functions.velocity.propagate_thriveline_z`
            Function that propagates z-scores along one segment:
            ``propagate(hop_correlations, start_z, z_thrive) -> xr.DataArray``.
        anchor_step : int, default 1
            Spacing between starting covariate anchors on the overlay grid.
        z_anchor_start, z_anchor_end : int, default -3 and 4
            Half-open range ``[z_anchor_start, z_anchor_end)`` of integer
            starting z-scores used when ``z_anchors`` is not provided.
        z_anchors : array-like of float, optional
            Explicit starting z-scores (e.g. ``norm.ppf(centiles)``). When
            given, ``z_anchor_start`` and ``z_anchor_end`` are ignored.
        covariate_range : tuple of int, optional
            ``(min, max)`` covariate bounds used to slice the correlation
            matrix and place grid anchors. When omitted, anchors span the
            covariate coordinates in the stored correlation matrix.

        Returns
        -------
        pd.DataFrame
            Long-form thriveline table with columns ``segment``, ``start_age``,
            ``start_z``, ``response_var``, ``offset``, ``X``, ``Z``, and ``Y``.
            Stored on the instance as :attr:`thrivelines`.
        """
        # Compute the matrix only if this object doesn't have one yet;
        # otherwise reuse the stored attribute without recomputing.
        if self.correlation_matrix is None:
            warnings.warn(
                "This ZGainScore has no correlation matrix yet; computing it "
                "from the reference data now.",
                UserWarning,
                stacklevel=2,
            )
            self.correlation_matrix = self.get_correlation_matrix()
        R = self.correlation_matrix
        # Step 1: propagate anchor segments in Z-space from the correlation matrix.
        # This bare name resolves to the imported velocity function (a module
        # global), not to this method, so it is not a recursive call.
        thrive_Z, thrive_X = compute_thrivelines(
            R,
            timepoint_diff=timepoint_diff,
            z_thrive=z_thrive,
            propagate=propagate,
            anchor_step=anchor_step,
            z_anchor_start=z_anchor_start,
            z_anchor_end=z_anchor_end,
            z_anchors=z_anchors,
            covariate_range=covariate_range,
        )
        # Step 2: map each (X, Z) point to response-scale Y via the normative model.
        thrive_Y = compute_thriveline_y(
            self.normative_model,
            thrive_Z,
            thrive_X,
            template=self.reference_data,
            covariate=self.covariate,
        )
        # Step 3: flatten to a long-form DataFrame for inspection and plotting.
        self.thrivelines = thrivelines_to_dataframe(thrive_Z, thrive_X, thrive_Y)
        return self.thrivelines

    def score(
        self,
        score_data: NormData,
        subject_id_col: str | None = None,
    ) -> xr.DataArray:
        """Compute the z-gain score for every subject in ``score_data``.

        Parameters
        ----------
        score_data : NormData
            Longitudinal cohort with at least two distinct visits per subject,
            numeric visit labels on the NormData object, and z-scores already
            computed.
        subject_id_col : str, optional
            Subject id column name override. Defaults to the value supplied
            at construction.

        Returns
        -------
        xr.DataArray
            One z-gain score per subject and response variable. The same array
            is also stored on the instance as :attr:`zgain` for later retrieval.
        """
        # Use the stored subject id name unless the caller overrides it.
        subject_id_col = subject_id_col or self.subject_id_col

        # Run checks
        self._check_is_predicted(score_data)
        self._check_is_longitudinal(score_data)

        # Read or estimate the correlation matrix.
        R = self.get_correlation_matrix()
        # Read the largest supported age index from that matrix.
        max_age = int(R[f"{self.covariate}_1"].max())

        # Read response-variable names for the output array.
        response_vars = [str(r) for r in score_data.response_vars.values]
        # Read subject ids so visits can be grouped per person.
        subject_ids = score_data.get_subject_ids()
        # Keep subjects in the same order as the input data.
        subjects = self._ordered_unique(subject_ids)
        # Map each subject id to its row in the output array.
        subject_index = {s: i for i, s in enumerate(subjects)}

        # Read the age value for every visit.
        ages = score_data.get_observation_column(self.covariate).astype(float)
        # Read the visit-order values used to sort trajectories.
        visits = score_data.get_visits()

        # Initialise an empty array filled with NaN to hold the scores.
        scores = np.full(
            (len(subjects), len(response_vars)),
            np.nan,
            dtype=float,
        )
        # Score one response variable at a time.
        for j, rv in enumerate(response_vars):
            # Select the correlation matrix for this response variable.
            R_rv = R.sel(response_vars=rv).values
            # Read the z-scores for this single response variable.
            z_rv = score_data.Z.sel(response_vars=rv).values
            # Score one subject at a time.
            for subject in subjects:
                # Find the visit rows for the current subject.
                idx = np.where(subject_ids == subject)[0]
                # Skip subjects that do not have enough visits.
                # TODO: Check if this allow the test data to have both
                # longitudinal and single-visit subjects as we remove
                # here the single-visit ones.
                if len(idx) < 2:
                    continue
                # Sort the visits into time order.
                ordered = idx[np.argsort(visits[idx])]
                # For longer trajectories, use the last observed step.
                # Keep only the last two visits for this score.
                obs_prev, obs_last = ordered[-2], ordered[-1]
                # Round and clip these two visits. We clip to the maximum
                # age supported by the correlation matrix.
                # TODO: Discuss with Johanna if it makes sense to round the
                # age here.
                age_prev = int(np.clip(round(ages[obs_prev]), 0, max_age))
                age_last = int(np.clip(round(ages[obs_last]), 0, max_age))

                # Read the specific correlation for this visit pair and keep
                # the denominator away from zero.
                r = float(
                    np.clip(
                        R_rv[age_prev, age_last],
                        -self.max_correlation,
                        self.max_correlation,
                    )
                )
                denominator = np.sqrt(1.0 - r**2)
                # Compute zgain
                scores[subject_index[subject], j] = (
                    z_rv[obs_last] - r * z_rv[obs_prev]
                ) / denominator

        # Store the result so it can be retrieved later via self.zgain, even
        # if the caller does not keep the returned value.
        self.zgain = xr.DataArray(
            scores,
            dims=("subjects", "response_vars"),
            coords={"subjects": subjects, "response_vars": response_vars},
            name="zgain",
        )
        return self.zgain
