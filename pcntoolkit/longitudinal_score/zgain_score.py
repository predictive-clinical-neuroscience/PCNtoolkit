from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from pcntoolkit.math_functions.velocity import get_correlation_matrix

from .longitudinal_score import LongitudinalScore

if TYPE_CHECKING:
    from pcntoolkit.dataio.norm_data import NormData
    from pcntoolkit.normative_model import NormativeModel


class ZGainScore(LongitudinalScore):
    """Generalised velocity-centile (z-gain) score (Bayer et al., 2026).

    The z-gain score asks whether a subject's z-score at a visit is unexpected
    given their z-score at an earlier visit. Using the estimated correlation
    ``r`` between z-scores at the two ages, the expected z-score at the later
    age is ``r·z_earlier``, with conditional standard deviation
    ``sqrt(1 − r²)``. The z-gain is the standardised deviation from that
    expectation::

        z_gain = (z_later − r·z_earlier) / sqrt(1 − r²)

    Unlike z-diff, this works for **any** regression model and **any** number of
    timepoints, and it does not require a warp.

    Reduction for more than two timepoints
    --------------------------------------
    :meth:`score` returns a single value per subject and response variable. When
    a subject has more than two visits, the score is computed for the **most
    recent transition** — i.e. the latest visit conditioned on the immediately
    preceding visit (ordered by ``timepoint_col``). Consecutive visits are the
    closest in age and therefore the best covered by the correlation matrix.
    Conditioning on the full visit history is provided separately by
    ``conditional_forecast`` (future work). For exactly two timepoints this is
    simply the later visit conditioned on the earlier one.

    Parameters
    ----------
    normative_model : NormativeModel
        A fitted normative model.
    norm_data : NormData
        Longitudinal data (z-scores predicted) used to estimate the correlation
        matrix.
    subject_id : str
        Subject id column name.
    bandwidth : int, default 5
        Age-offset range (in years) within which correlations are estimated
        directly; larger offsets are interpolated.
    covariate : str, default "age"
        Covariate that indexes the correlation matrix.
    """

    def __init__(
        self,
        normative_model: "NormativeModel",
        norm_data: "NormData",
        subject_id: str,
        bandwidth: int = 5,
        covariate: str = "age",
        max_correlation: float = 0.99,
    ):
        super().__init__(normative_model, norm_data, subject_id)
        self.bandwidth = bandwidth
        self.covariate = covariate
        # Correlations are clipped to +/- max_correlation when forming the
        # conditional variance, so z-gain stays finite even when estimated
        # correlations saturate to +/-1 for sparsely sampled age pairs.
        self.max_correlation = max_correlation
        # (response_vars, age_1, age_2) correlation matrix, filled lazily by
        # compute_correlation_matrix() on first use.
        self.correlation_matrix: xr.DataArray | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def compute_correlation_matrix(self) -> xr.DataArray:
        """Estimate (and cache) the z-score correlation matrix from ``norm_data``."""
        if self.correlation_matrix is None:
            self._check_is_predicted(self.norm_data)
            self.correlation_matrix = get_correlation_matrix(self.norm_data, self.bandwidth, self.covariate)
        return self.correlation_matrix

    def score(self, data: "NormData", subject_id: str | None = None, timepoint_col: str = "visit") -> xr.DataArray:
        """Compute the z-gain score for every subject in ``data``.

        Parameters
        ----------
        data : NormData
            Longitudinal data to score, with predictions/z-scores already
            computed (``model.predict(data)``).
        subject_id : str, optional
            Subject id column name (kept for API symmetry; subject ids are read
            from the ``NormData`` ``subject_ids`` field). Defaults to the value
            passed at construction.
        timepoint_col : str, default "visit"
            Column (batch effect or covariate) used to order visits.

        Returns
        -------
        xr.DataArray
            ``(subjects, response_vars)`` z-gain scores. Subjects with fewer
            than two timepoints are ``NaN``.
        """
        subject_id = subject_id or self.subject_id
        self._check_is_predicted(data)
        self._check_is_longitudinal(data)

        R = self.compute_correlation_matrix()
        max_age = int(R[f"{self.covariate}_1"].max())

        response_vars = [str(r) for r in data.response_vars.values]
        subject_ids = self._get_subject_ids(data)
        subjects = self._ordered_unique(subject_ids)
        subject_index = {s: i for i, s in enumerate(subjects)}

        ages = self._get_observation_column(data, self.covariate).astype(float)
        timepoints = self._get_timepoint_values(data, timepoint_col)

        scores = np.full((len(subjects), len(response_vars)), np.nan, dtype=float)
        for j, rv in enumerate(response_vars):
            R_rv = R.sel(response_vars=rv).values  # (n_ages, n_ages), indexed by integer age
            z_rv = data.Z.sel(response_vars=rv).values
            for subject in subjects:
                idx = np.where(subject_ids == subject)[0]
                if len(idx) < 2:
                    continue
                ordered = idx[np.argsort(timepoints[idx])]
                # Most recent transition: previous -> latest visit.
                i_prev, i_last = ordered[-2], ordered[-1]
                a_prev = int(np.clip(round(ages[i_prev]), 0, max_age))
                a_last = int(np.clip(round(ages[i_last]), 0, max_age))
                # Clip away from +/-1 so the conditional variance stays well-defined.
                r = float(np.clip(R_rv[a_prev, a_last], -self.max_correlation, self.max_correlation))
                denom = np.sqrt(1.0 - r**2)
                scores[subject_index[subject], j] = (z_rv[i_last] - r * z_rv[i_prev]) / denom

        return xr.DataArray(
            scores,
            dims=("subjects", "response_vars"),
            coords={"subjects": subjects, "response_vars": response_vars},
            name="zgain",
        )
