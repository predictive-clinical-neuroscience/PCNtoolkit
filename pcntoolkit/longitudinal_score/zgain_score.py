from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from pcntoolkit.math_functions.correlation_matrix import CorrelationMatrix
from pcntoolkit.math_functions.velocity import (
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
    correlation_matrix : CorrelationMatrix
        Age-to-age z-score correlations, estimated with
        :meth:`~pcntoolkit.math_functions.correlation_matrix.CorrelationMatrix.compute`
        or loaded from a saved file.

    Attributes
    ----------
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
        correlation_matrix: CorrelationMatrix,
    ):
        # Reuse the shared setup from the base class.
        super().__init__(normative_model)

        self.correlation_matrix = correlation_matrix
        self.covariate = correlation_matrix.covariate

        # Hold the most recent z-gain scores; filled in by score().
        self.zgain: xr.DataArray | None = None

        # Hold the most recent thrivelines; filled in by get_thrivelines().
        self.thrivelines: pd.DataFrame | None = None

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
        propagates thrivelines in Z-space, then maps them to response-scale Y
        via :attr:`normative_model`. Covariates other than the thrive covariate
        are fixed to the midpoint of each covariate's range in the normative
        model, the same rule
        :func:`~pcntoolkit.util.plotter.plot_centiles` uses when it is given no
        scatter data. Batch effects use the first allowed level from the
        normative model.

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
        # Step 1: propagate anchor segments in Z-space from the correlation matrix.
        # This bare name resolves to the imported velocity function (a module
        # global), not to this method, so it is not a recursive call.
        thrive_Z, thrive_X = compute_thrivelines(
            self.correlation_matrix.matrix,
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
            Subject id column name. Kept for backwards compatibility; the
            subject ids are read from ``score_data`` itself.

        Returns
        -------
        xr.DataArray
            One z-gain score per subject and response variable. The same array
            is also stored on the instance as :attr:`zgain` for later retrieval.
        """
        # Run checks
        self._check_is_predicted(score_data)
        self._check_is_longitudinal(score_data)

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
                # TODO: Discuss with Johanna if it makes sense to round the
                # age here.
                age_prev = int(round(ages[obs_prev]))
                age_last = int(round(ages[obs_last]))

                # Read the correlation for this visit pair. get() clamps the
                # ages to the matrix and keeps the denominator away from zero.
                r = self.correlation_matrix.get(rv, age_prev, age_last)
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
