from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

# Import BLR because z-diff is defined only for BLR models.
from pcntoolkit.regression_model.blr import BLR

from .longitudinal_score import LongitudinalScore

# Import the heavy classes only during type checking.
if TYPE_CHECKING:
    from pcntoolkit.dataio.norm_data import NormData
    from pcntoolkit.normative_model import NormativeModel


class ZDiffScore(LongitudinalScore):
    """Two-visit z-diff score (Rehák Bučková et al., 2025).

    The z-diff score asks whether the change between two visits is larger or
    smaller than expected under a fitted BLR normative model.

        Δr_target = (y₂ - ŷ₂) - (y₁ - ŷ₁)

    The toolkit scales that change by the spread of changes learned
    from ``reference_data`` (can be the healthy subjects)::

        z_diff = Δr_target / SD(Δr_reference)

        where SD(Δr_reference) = sqrt(2σ²(1−ρ)) if the residuals are
        stationary with variance σ² and visit-to-visit correlation ρ.

    This score is intended for BLR-based models and subjects with at most two
    visits.

    Attributes
    ----------
    zdiff : xr.DataArray | None
        The most recent z-diff scores produced by :meth:`score`. ``None`` until
        :meth:`score` has been called at least once. It stores the same
        ``xr.DataArray`` that :meth:`score` returns, so the result can be
        retrieved later even if the return value was not saved.
    """

    def __init__(
        self,
        normative_model: NormativeModel,
        reference_data: NormData,
        subject_id_col: str, # TODO: Are these subject_id_col necessary as a keyword argument? Does the user need to specify that?
    ):
        # Reuse the shared setup from the base class.
        super().__init__(normative_model, reference_data, subject_id_col)
        # z-diff is defined only for BLR models.
        self._check_model_is_blr(normative_model)

        # Hold the most recent z-diff scores; filled in by score().
        self.zdiff: xr.DataArray | None = None

    def score(
        self,
        score_data: NormData,
        subject_id_col: str | None = None,
    ) -> xr.DataArray:
        """Compute the z-diff score for every subject in ``score_data``.

        Parameters
        ----------
        score_data : NormData
            Longitudinal cohort with exactly two visits per subject, numeric
            visit labels on the NormData object, and predictions already
            computed.
        subject_id_col : str, optional
            Subject id column name override. Defaults to the value supplied
            at construction.

        Returns
        -------
        xr.DataArray
            One z-diff score per subject and response variable. The same array
            is also stored on the instance as :attr:`zdiff` for later retrieval.
        """
        # Use the stored subject id name unless the caller overrides it.
        subject_id_col = subject_id_col or self.subject_id_col

        # Run checks first for both the reference and score data.
        self._check_is_predicted(self.reference_data)
        self._check_is_longitudinal(self.reference_data)
        self._check_at_most_two_visits(self.reference_data)
        self._check_is_predicted(score_data)
        self._check_is_longitudinal(score_data)
        self._check_at_most_two_visits(score_data)

        # Read response-variable names for the output array.
        response_vars = [str(r) for r in score_data.response_vars.values]
        # Keep subjects in the same order as the input data.
        subjects = self._ordered_unique(self._get_subject_ids(score_data))
        # Map each subject id to its row in the output array.
        subject_index = {s: i for i, s in enumerate(subjects)}

        # Initialise an empty array filled with NaN to hold the scores.
        scores = np.full(
            (len(subjects), len(response_vars)),
            np.nan,
            dtype=float,
        )

        # Score one response variable at a time.
        for j, rv in enumerate(response_vars):
            # Learn the typical size of expected change from the reference
            # cohort (reference_data).
            delta_reference = self._compute_residual_change(
                self.reference_data, rv
            )
            # Convert the subject-level changes into a numeric vector.
            delta_reference_values = np.fromiter(
                delta_reference.values(),
                dtype=float,
            )
            # Estimate the spread of expected changes for this response.
            # Mathematically this is SD(Δr_reference)
            denominator = np.sqrt(float(np.mean(delta_reference_values**2)))
            # Reject the degenerate case of no expected change at all.
            if np.isclose(denominator, 0.0):
                raise ValueError(
                    f"Cannot estimate denominator for '{rv}': "
                    "reference_data has zero residual-change variability."
                )

            # Compute the change for each target subject (score_data).
            # Mathematically this Δr_target.
            deltas_target = self._compute_residual_change(
                score_data,
                rv,
            )
            # Compute zdiff
            for subject, delta_target in deltas_target.items():
                scores[subject_index[subject], j] = delta_target / denominator

        # Store the result so it can be retrieved later via self.zdiff, even
        # if the caller does not keep the returned value.
        self.zdiff = xr.DataArray(
            scores,
            dims=("subjects", "response_vars"),
            coords={"subjects": subjects, "response_vars": response_vars},
            name="zdiff",
        )
        return self.zdiff

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _compute_residual_change(
        self,
        data: NormData,
        responsevar: str,
    ) -> dict[object, float]:
        """Compute the residual change from the first visit to the second for
        each subject."""
        # Compute a residual value for every visit.
        residuals = self._compute_residual(data, responsevar)

        # Read subject ids so visits can be grouped per person.
        subject_ids = self._get_subject_ids(data)
        # Read the visit-order values used to sort repeated measures.
        visits = self._get_visits(data)

        # Collect one change value per subject.
        deltas: dict[object, float] = {}
        for subject in self._ordered_unique(subject_ids):
            # Find the visit rows for the current subject.
            idx = np.where(subject_ids == subject)[0]
            # Skip subjects that do not have enough visits. TODO: Check if this allow the test data to have both longitudinal and single-visit subjects as we remove here the single-visit ones.
            if len(idx) < 2:
                continue
            # Put the two visits into chronological order.
            ordered = idx[np.argsort(visits[idx])]
            # Read the residual at visit 1 and visit 2.
            r1, r2 = residuals[ordered[0]], residuals[ordered[1]]
            # Store the change from the first visit to the second.
            deltas[subject] = r2 - r1
        # Return the changes for all subjects.
        return deltas

    def _compute_residual(
        self,
        data: NormData,
        responsevar: str,
    ) -> np.ndarray:
        """Compute the residual for each visit."""
        # Get the fitted BLR model for the selected response variable.
        blr = self.normative_model[responsevar]
        # Read the observed measurements for this region.
        y = np.asarray(
            data.Y.sel(response_vars=responsevar).values,
            dtype=float,
        )
        # Read the predicted measurements for this region.
        yhat = np.asarray(
            data.Yhat.sel(response_vars=responsevar).values,
            dtype=float,
        )

        # If the BLR used a warp, compare values on that fitted scale.
        if getattr(blr, "warp", None) is not None:
            # Compare observed and predicted values on the same scale that the
            # fitted BLR used internally.
            # Get the output scaler paired with this response variable.
            outscaler = self.normative_model.outscalers[responsevar]
            # Transform and warp the observed values.
            y = blr.warp.f(outscaler.transform(y), blr.gamma)
            # Transform and warp the predicted values.
            yhat = blr.warp.f(outscaler.transform(yhat), blr.gamma)

        # Return the residual that z-diff uses at each visit.
        return y - yhat

    # ------------------------------------------------------------------ #
    # Validation functions
    # ------------------------------------------------------------------ #
    def _check_at_most_two_visits(self, data: NormData) -> None:
        # Read subject ids so visit counts can be checked.
        ids = self._get_subject_ids(data)
        # Count how many visits each subject contributes.
        _, counts = np.unique(ids, return_counts=True)
        # Reject any subject with more than two visits.
        if np.any(counts > 2):
            # Tell the user to switch to z-gain for longer trajectories.
            raise ValueError(
                "ZDiffScore supports at most two visits per subject. "
                "Some subjects have more than two visits. You can use ZGainScore "
                "for three or more visits."
            )

    @staticmethod
    def _check_model_is_blr(normative_model: NormativeModel) -> None:
        # Read the template model stored inside the normative wrapper.
        template = getattr(normative_model, "template_regression_model", None)
        # Reject models that are not BLR-based.
        if not isinstance(template, BLR):
            # Explain that z-diff is only defined for BLR models here.
            raise ValueError(
                "ZDiffScore requires a BLR (or warped BLR) normative model. "
                "Use ZGainScore for other regression models."
            )
