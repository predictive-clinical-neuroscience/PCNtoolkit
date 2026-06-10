from __future__ import annotations

# Import typing-only helpers to avoid runtime circular imports.
from typing import TYPE_CHECKING

# Import NumPy for indexing and score calculations.
import numpy as np
# Import xarray for labeled correlation matrices and outputs.
import xarray as xr

# Import the helper that learns age-to-age z-score correlations.
from pcntoolkit.math_functions.velocity import get_correlation_matrix

# Import the shared longitudinal-score base class.
from .longitudinal_score import LongitudinalScore

# Import toolkit types only for static type checking.
if TYPE_CHECKING:
    # Import the labeled data container used during scoring.
    from pcntoolkit.dataio.norm_data import NormData
    # Import the normative-model wrapper used by the score.
    from pcntoolkit.normative_model import NormativeModel


class ZGainScore(LongitudinalScore):
    """Generalised velocity-centile (z-gain) score (Bayer et al., 2026).

    The z-gain score asks whether a subject's later z-score is surprising once
    their earlier z-score is known. If the typical age-to-age correlation is
    ``r``, the score is

    ::

        z_gain = (z_later − r·z_earlier) / sqrt(1 − r²)

    It works with any regression model and any number of visits. When more than
    two visits are available, the score uses the most recent transition.

    Parameters
    ----------
    normative_model : NormativeModel
        A fitted normative model.
    reference_data : NormData
        Longitudinal reference cohort used to learn how z-scores typically
        persist across ages. Predictions must already be available.
    subject_id_col : str
        Name of the column that identifies subjects inside both
        ``reference_data`` and the ``test_data`` passed to :meth:`score`.
    bandwidth : int, default 5
        Age-offset range, in years, for direct correlation estimates.
    covariate : str, default "age"
        Covariate that indexes the correlation matrix.
    max_correlation : float, default 0.99
        Upper bound used to keep the denominator safely away from zero.
    """

    def __init__(
        self,
        normative_model: "NormativeModel",
        reference_data: "NormData",
        subject_id_col: str,
        bandwidth: int = 5,
        covariate: str = "age",
        max_correlation: float = 0.99,
    ):
        # Reuse the shared setup from the base class.
        super().__init__(normative_model, reference_data, subject_id_col)
        # Keep the clipping threshold inside a mathematically safe range.
        if not 0.0 < max_correlation < 1.0:
            # Explain that the denominator must stay strictly positive.
            raise ValueError(
                "max_correlation must be strictly between 0 and 1."
            )
        # Store the bandwidth used when learning correlations.
        self.bandwidth = bandwidth
        # Store the covariate that indexes the correlation matrix.
        self.covariate = covariate
        # Keep the denominator finite when the estimated correlation is nearly
        # perfect for a sparse age pair.
        # Store the clipping threshold for very large correlations.
        self.max_correlation = max_correlation
        # Cache the learned age-to-age correlation matrix after first use.
        # Start with no cached matrix so it can be learned lazily.
        self.correlation_matrix: xr.DataArray | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def compute_correlation_matrix(self) -> xr.DataArray:
        """Estimate and cache the age-to-age z-score correlation matrix."""
        # Learn the matrix only once and then reuse it.
        if self.correlation_matrix is None:
            # Make sure the reference cohort already has predictions.
            self._check_is_predicted(self.reference_data)
            # Make sure the reference cohort really has repeated visits.
            self._check_is_longitudinal(self.reference_data)
            # Estimate how strongly z-scores persist across ages.
            self.correlation_matrix = get_correlation_matrix(
                self.reference_data, self.bandwidth, self.covariate
            )
        # Return the cached or newly estimated matrix.
        return self.correlation_matrix

    def score(
        self,
        test_data: "NormData",
        subject_id_col: str | None = None,
        timepoint_col: str = "visit",
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
            Column used to order visits.

        Returns
        -------
        xr.DataArray
            One z-gain score per subject and response variable.
        """
        # Use the stored subject id name unless the caller overrides it.
        subject_id_col = subject_id_col or self.subject_id_col
        # Make sure the scored cohort already has z-scores and predictions.
        self._check_is_predicted(test_data)
        # Make sure the scored cohort has repeated visits.
        self._check_is_longitudinal(test_data)

        # The reference cohort tells us how strongly z-scores usually persist
        # from one age to the next.
        # Read or estimate the age-to-age correlation matrix.
        R = self.compute_correlation_matrix()
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

        # Prepare an output array filled with missing values by default.
        scores = np.full((len(subjects), len(response_vars)), np.nan, dtype=float)
        # Score one response variable at a time.
        for j, rv in enumerate(response_vars):
            # Select the correlation matrix for this single response variable.
            R_rv = R.sel(response_vars=rv).values
            # Read the z-scores for this single response variable.
            z_rv = test_data.Z.sel(response_vars=rv).values
            # Score one subject at a time.
            for subject in subjects:
                # Find the visit rows for the current subject.
                idx = np.where(subject_ids == subject)[0]
                # Skip subjects that do not have enough visits.
                if len(idx) < 2:
                    continue
                # Sort the visits into time order.
                ordered = idx[np.argsort(timepoints[idx])]
                # For longer trajectories, use the last observed step.
                # Keep only the last two visits for this score.
                i_prev, i_last = ordered[-2], ordered[-1]
                # Round and clip the earlier age to valid matrix bounds.
                a_prev = int(np.clip(round(ages[i_prev]), 0, max_age))
                # Round and clip the later age to valid matrix bounds.
                a_last = int(np.clip(round(ages[i_last]), 0, max_age))
                # Keep the denominator away from zero.
                # Read the age-specific correlation for this visit pair.
                r = float(
                    np.clip(
                        R_rv[a_prev, a_last],
                        -self.max_correlation,
                        self.max_correlation,
                    )
                )
                # Convert that correlation into the conditional spread.
                denom = np.sqrt(1.0 - r**2)
                # Standardize the later z-score given the earlier z-score.
                scores[subject_index[subject], j] = (z_rv[i_last] - r * z_rv[i_prev]) / denom

        # Return labeled scores so subjects and regions stay identifiable.
        return xr.DataArray(
            scores,
            dims=("subjects", "response_vars"),
            coords={"subjects": subjects, "response_vars": response_vars},
            name="zgain",
        )
