from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from pcntoolkit.math_functions.velocity import (
    compute_thriveline_y,
    compute_thriveline_z,
    thrivelines_to_dataframe,
)

from .correlation_matrix import CorrelationMatrix
from .longitudinal_score import LongitudinalScore

# Import the heavy classes only during type checking.
if TYPE_CHECKING:
    from pcntoolkit.dataio.norm_data import NormData
    from pcntoolkit.normative_model import NormativeModel


class VelocityModel(LongitudinalScore):
    """Velocity model introduced by Bayer et al., 2026 (preprint).

    It works with any regression model and any number of visits. With more
    than two visits, only the last two are used.
    TODO: Using on the full visit history and not only the most recent
    transition can be done with ``conditional_forecast()``

    Parameters
    ----------
    normative_model : NormativeModel
        A fitted normative model.
    correlation_matrix : CorrelationMatrix
        How strongly z-scores at two timepoints correlate, from
        ``CorrelationMatrix.compute`` or ``CorrelationMatrix.load``.

    Attributes
    ----------
    zgain : xr.DataArray | None
        Last result of ``score``, or ``None`` before the first call.
    thrivelines : pd.DataFrame | None
        Last result of ``compute_thrivelines``, or ``None`` before the first call.
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

        # Hold the most recent thrivelines; filled in by compute_thrivelines().
        self.thrivelines: pd.DataFrame | None = None

    def compute_thrivelines(
        self,
        *,
        zgain_threshold: float = -1.96,
        thriveline_span: int = 1,
        thriveline_spacing: int = 1,
        centiles: list[float] | np.ndarray | None = None,
        covariate_range: tuple[int, int] | None = None,
        save_path: str | PathLike[str] | None = None,
    ) -> pd.DataFrame:
        """Compute thrivelines to draw on the centile chart.

        A thriveline is a short line that starts on a centile curve and
        shows the largest change a person starting there can have in
        ``thriveline_span`` years and still be within the expected range.
        Someone who ends up beyond the line has changed more than expected.
        Draw the result with ``plot_thrivelines``.

        By default, lines use the model's first batch effect level, and the rest of
        the model's covariates are held at the middle of their range.

        Parameters
        ----------
        zgain_threshold : float, default -1.96
            z-gain each line follows, which sets its slope. It represents
            largest expected change a subject can have in ``thriveline_span`` years.
        thriveline_span : int, default 1
            How far each thriveline stretches along the x-axis.
        thriveline_spacing : int, default 1
            How far apart neighbouring thrivelines are along the x-axis.
        centiles : array-like of float, optional
            Centile curves the lines start on, between 0 and 1.
            Defaults to ``[0.05, 0.25, 0.5, 0.75, 0.95]``.
        covariate_range : tuple of int, optional
            ``(min, max)`` ages to draw lines for, e.g. ``(20, 60)``. Defaults
            to the ages the normative model was trained on.
        save_path : str | PathLike[str] | None, optional
            If set, also save the table to this CSV file.

        Returns
        -------
        pd.DataFrame
            Example:

                segment  response_var  offset   X       Z     Y
                0        lh_thickness  0       20   0.000  2.71
                0        lh_thickness  1       21  -1.176  2.62

            ``segment`` numbers the lines, ``offset`` is 0 at the start and
            ``thriveline_span`` at the end, ``X`` is age, ``Z`` the z-score and
            ``Y`` the brain measure. Lines with a missing correlation are left
            out, with a warning. Also stored as ``self.thrivelines``.

        Raises
        ------
        ValueError
            If ``thriveline_span`` is not positive, or if a centile is
            not strictly between 0 and 1.
        """
        if covariate_range is None:
            covariate_range = self._default_covariate_range()
        # Step 1: compute the start and end z-score of every line.
        thrive_Z, thrive_X = compute_thriveline_z(
            self.correlation_matrix.matrix,
            zgain_threshold=zgain_threshold,
            thriveline_span=thriveline_span,
            thriveline_spacing=thriveline_spacing,
            centiles=centiles,
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
        # Lets plot_thrivelines put the right covariate on the x-axis.
        self.thrivelines.attrs["covariate"] = self.covariate
        if save_path is not None:
            self._save_dataframe_csv(self.thrivelines, save_path)
        return self.thrivelines

    def _default_covariate_range(self) -> tuple[int, int]:
        """Pick the covariate range of the user's normative model to plot thrivelines
        and not the correlation matrix's full range.

        Useful when the user loads a correlation matrix that has a wider time range than
        the user's normative model was trained on.

        Returns
        -------
        tuple of int
            ``(min, max)`` timepoints the normative model was trained on, limited to
            the timepoints the correlation matrix covers.

        Raises
        ------
        ValueError
            If that range does not overlap at all with the time range of the
            correlation matrix.
        """
        matrix_ages = self.correlation_matrix.matrix[f"{self.covariate}_1"].values
        matrix_min, matrix_max = int(matrix_ages.min()), int(matrix_ages.max())

        model_ranges = self.normative_model.covariate_ranges or {}
        if self.covariate not in model_ranges:
            # Models saved without covariate ranges: keep the old behaviour.
            return matrix_min, matrix_max
        # Widen to whole years, as the matrix is indexed by integer ages.
        low = int(np.floor(model_ranges[self.covariate]["min"]))
        high = int(np.ceil(model_ranges[self.covariate]["max"]))

        if max(low, matrix_min) >= min(high, matrix_max):
            raise ValueError(
                f"The normative model's {self.covariate} range ({low}-{high}) does not "
                f"overlap the correlation matrix ({matrix_min}-{matrix_max}), so "
                f"there are no {self.covariate} values to draw thrivelines for. Use "
                f"a correlation matrix that covers the {self.covariate} values of "
                "the normative model."
            )
        return max(low, matrix_min), min(high, matrix_max)

    def score(
        self,
        score_data: NormData,
        save_path: str | PathLike[str] | None = None,
    ) -> xr.DataArray:
        """Compute one z-gain per subject and response variable.

        Near 0: the person changed as expected. Below -1.96: an unusually
        large decline. Above +1.96: an unusually large rise.

        Parameters
        ----------
        score_data : NormData
            Subjects to score, with z-scores already predicted and visit
            numbers set. Subjects with a single visit get NaN.
        save_path : str | PathLike[str] | None, optional
            If set, also save the scores to this CSV file::

                subjects  lh_thickness  rh_thickness
                sub-01            0.42         -1.10
                sub-02           -0.35          0.87

        Returns
        -------
        xr.DataArray
            z-gain with dimensions ``(subjects, response_vars)``. Also stored
            as ``self.zgain``.
        """
        # Run checks
        self._check_is_predicted(score_data, required=("Z",))
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

                # Read the correlation for this visit pair. get_correlation() clamps the
                # ages to the matrix and keeps the denominator away from zero.
                r = self.correlation_matrix.get_correlation(rv, age_prev, age_last)
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
        if save_path is not None:
            self._save_scores_csv(self.zgain, save_path)
        return self.zgain
