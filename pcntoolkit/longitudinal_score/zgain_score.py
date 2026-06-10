from __future__ import annotations

from typing import TYPE_CHECKING

from dask.base import compute
import numpy as np
import xarray as xr

from pcntoolkit.math_functions.velocity import compute_correlation_matrix

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
        ``reference_data`` and the ``test_data``.
    bandwidth : int, default 5
        Age-offset range, in years, for direct correlation estimates.
    covariate : str, default "age"
        The dimension for which the correlations are calculated in the
        correlation matrix.
    max_correlation : float, default 0.99
        Upper bound used to keep the correlation away from 1 and 
        the denominator away from zero. 
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

    def get_correlation_matrix(self) -> xr.DataArray:
        """Estimate and cache the z-score correlation matrix. It computes
        the matrix once and then reuses it for all subsequent calls."""
        if self.correlation_matrix is None:
            self.correlation_matrix = compute_correlation_matrix(
                self.reference_data, self.bandwidth, self.covariate
            )
        return self.correlation_matrix

    def score(
        self,
        test_data: NormData,
        subject_id_col: str | None = None,
        timepoint_col: str = "visit",  # TODO: The LNM_data.csv uses visits to group longitudinal data in a LONG DATAFRAME. Other datasets might use a WIDE DATAFRAME with multiple columns visit_1, visit_2 etc. How can we handle that?
    ) -> xr.DataArray:
        """Compute the z-gain score for every subject in ``test_data``.

        Parameters
        ----------
        test_data : NormData
            Longitudinal cohort with at least two visits per subject and
            z-scores already computed.
        subject_id_col : str, optional
            Subject id column name override. Defaults to the value supplied
            at construction.
        timepoint_col : str, default "visit"
            Column used to order multiple time points/visits.

        Returns
        -------
        xr.DataArray
            One z-gain score per subject and response variable.
        """
        # Use the stored subject id name unless the caller overrides it.
        subject_id_col = subject_id_col or self.subject_id_col

        # Run checks
        self._check_is_predicted(test_data)
        self._check_is_longitudinal(test_data)

        # Read or estimate the correlation matrix.
        R = self.get_correlation_matrix()
        # Read the largest supported age index from that matrix.
        max_age = int(R[f"{self.covariate}_1"].max())

        # Read response-variable names for the output array.
        response_vars = [str(r) for r in test_data.response_vars.values]
        # Read subject ids so visits can be grouped per person.
        subject_ids = self._get_subject_ids(test_data)
        # Keep subjects in the same order as the input data.
        subjects = self._ordered_unique(subject_ids)
        # Map each subject id to its row in the output array.
        subject_index = {s: i for i, s in enumerate(subjects)}

        # Read the age value for every visit.
        ages = self._get_observation_column(test_data, self.covariate).astype(
            float
        )
        # Read the visit-order values used to sort trajectories.
        timepoints = self._get_timepoint_values(test_data, timepoint_col)

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
            z_rv = test_data.Z.sel(response_vars=rv).values
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
                ordered = idx[np.argsort(timepoints[idx])]
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

        return xr.DataArray(
            scores,
            dims=("subjects", "response_vars"),
            coords={"subjects": subjects, "response_vars": response_vars},
            name="zgain",
        )
