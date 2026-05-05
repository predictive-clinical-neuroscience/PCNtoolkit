from typing import List, Tuple

import numpy as np
import xarray as xr
from scipy import stats  # type: ignore
from sklearn.metrics import (explained_variance_score, r2_score,
                             mean_absolute_percentage_error)

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.util.data_utils import iter_batch_combinations
from pcntoolkit.util.output import Output, Warnings


class Evaluator:
    """
    A class for evaluating normative model predictions.

    This class implements various statistics to assess the quality of
    normative model predictions, including correlation coefficients, error metrics,
    and normality tests.

    Attributes
    ----------
    response_vars : List[str]
        List of response variables to evaluate
    """

    def __init__(self) -> None:
        """Initialize the Evaluator."""
        self.response_vars: List[str] = []

    def evaluate(self, data: NormData, statistics: List[str] = []) -> NormData:
        """
        Evaluate model predictions using multiple statistics.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values, and yhat

        Returns
        -------
        NormData
            Data container updated with evaluation statistics
        """
        # data["Yhat"] = data.centiles.sel(centile=0.5, method="nearest")
        assert "Yhat" in data.data_vars, "Yhat must be computed before evaluation"
        all_statistics = [
            "Rho", "Rho_p", "R2", "RMSE", "SMSE",
            "MSLL", "MLL", "ShapiroW", "MACE", "MAPE", "EXPV",
            "Skew", "Kurt",
        ]
        if statistics:
            self.statistics = [m for m in all_statistics if m in statistics]

        else:
            self.statistics = all_statistics
        if "Rho" in self.statistics and "Rho_p" not in self.statistics:
            self.statistics.append("Rho_p")
        self.create_statistics_group(data)
        if "ShapiroW" in self.statistics:
            self.evaluate_shapiro_w(data)
        if "R2" in self.statistics:
            self.evaluate_R2(data)
        if "Rho" in self.statistics:
            self.evaluate_rho(data)
        if "RMSE" in self.statistics:
            self.evaluate_rmse(data)
        if "SMSE" in self.statistics:
            self.evaluate_smse(data)
        if "MSLL" in self.statistics:
            self.evaluate_msll(data)
        if "MLL" in self.statistics:
            self.evaluate_mll(data)
        if "MACE" in self.statistics:
            self.evaluate_mace(data)
        if "MAPE" in self.statistics:
            self.evaluate_mape(data)
        if "EXPV" in self.statistics:
            self.evaluate_expv(data)
        if "Skew" in self.statistics:
            # Evaluate skewness of the z-score distribution
            self.evaluate_skew(data)
        if "Kurt" in self.statistics:
            # Evaluate excess kurtosis of the z-score distribution
            self.evaluate_kurt(data)
        return data

    def create_statistics_group(self, data: NormData) -> None:
        """
        Create a statistics group in the data container.

        Parameters
        ----------
        data : NormData
            Data container to add statistics group to
        """
        self.statistics = sorted(self.statistics)
        data["statistics"] = xr.DataArray(
            np.nan * np.ones((len(data.response_var_list),
                             len(self.statistics))),
            dims=("response_vars", "statistic"),
            coords={
                "response_vars": data.response_var_list,
                "statistic": self.statistics,
            },
        )

    def evaluate_rho(self, data: NormData) -> None:
        """
        Evaluate Spearman's rank correlation coefficient.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values
        """
        for responsevar in data.response_var_list:
            resp_predict_data = data.sel(response_vars=responsevar)
            rho, p_rho = self._evaluate_rho(resp_predict_data)
            data.statistics.loc[{
                "response_vars": responsevar, "statistic": "Rho"}] = float(rho)
            data.statistics.loc[{"response_vars": responsevar,
                                 "statistic": "Rho_p"}] = float(p_rho)

    def evaluate_R2(self, data: NormData) -> None:
        """
        Evaluate R2 for model predictions.
        """
        for responsevar in data.response_var_list:
            resp_predict_data = data.sel(response_vars=responsevar)
            r2 = self._evaluate_R2(resp_predict_data)
            data.statistics.loc[{
                "response_vars": responsevar, "statistic": "R2"}] = r2

    def evaluate_rmse(self, data: NormData) -> None:
        """
        Evaluate Root Mean Square Error (RMSE) for model predictions.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values. Must contain 'y' and 'Yhat' variables.
        """
        for responsevar in data.response_var_list:
            resp_predict_data = data.sel(response_vars=responsevar)
            rmse = self._evaluate_rmse(resp_predict_data)
            data.statistics.loc[{
                "response_vars": responsevar, "statistic": "RMSE"}] = rmse

    def evaluate_smse(self, data: NormData) -> None:
        """
        Evaluate Standardized Mean Square Error (SMSE) for model predictions.

        SMSE normalizes the mean squared error by the variance of the target variable,
        making it scale-independent.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values. Must contain 'y' and 'Yhat' variables.
        """
        for responsevar in data.response_var_list:
            resp_predict_data = data.sel(response_vars=responsevar)
            smse = self._evaluate_smse(resp_predict_data)
            data.statistics.loc[{
                "response_vars": responsevar, "statistic": "SMSE"}] = smse

    def evaluate_expv(self, data: NormData) -> None:
        """
        Evaluate Explained Variance score for model predictions.

        The explained variance score statistics the proportion of variance in the target variable
        that is predictable from the input features.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values. Must contain 'y' and 'Yhat' variables.
        """
        for responsevar in data.response_var_list:
            resp_predict_data = data.sel(response_vars=responsevar)
            expv = self._evaluate_expv(resp_predict_data)
            data.statistics.loc[{
                "response_vars": responsevar, "statistic": "EXPV"}] = expv

    def evaluate_msll(self, data: NormData) -> None:
        """
        Evaluate Mean Standardized Log Loss (MSLL) for model predictions.

        MSLL compares the log loss of the model to that of a simple baseline predictor
        that always predicts the mean of the training data.

        MSLL = MLL_model - MLL_baseline

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values. Must contain 'y', 'Yhat',
            and standard deviation predictions.
        """
        for responsevar in data.response_var_list:
            resp_predict_data = data.sel(response_vars=responsevar)
            msll = self._evaluate_msll(resp_predict_data)
            data.statistics.loc[{
                "response_vars": responsevar, "statistic": "MSLL"}] = msll

    def evaluate_mll(self, data: NormData) -> None:
        """
        Evaluate Mean Log Loss (MLL) for model predictions.

        MLL measures the probabilistic accuracy of the model's predictions.

        Note: In earlier PCNtoolkit releases, this metric was called `NLL`
        (Negative Log Likelihood). It is now named `MLL` to match the
        literature and avoid confusion with the different `NLL` used internally
        for BLR hyperparameter estimation.

        Parameters
        ----------
        data : NormData
                Data container with predictions and actual values. Must contain
                'logp' values for the evaluated response variable.
        """
        for responsevar in data.response_var_list:
            resp_predict_data = data.sel(response_vars=responsevar)
            mll = self._evaluate_mll(resp_predict_data)
            data.statistics.loc[{
                "response_vars": responsevar, "statistic": "MLL"}] = mll

    def evaluate_bic(self, data: NormData) -> None:
        """
        Evaluate Bayesian Information Criterion (BIC) for model predictions.

        BIC is a criterion for model selection that statistics the trade-off between
        model fit and complexity.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values. Must contain 'y' and 'Yhat' variables.
        """
        for responsevar in data.response_var_list:
            resp_predict_data = data.sel(response_vars=responsevar)
            bic = self._evaluate_bic(resp_predict_data)
            self.prepare(responsevar)
            data.statistics.loc[{
                "response_vars": responsevar, "statistic": "BIC"}] = bic
            self.reset()

    def evaluate_shapiro_w(self, data: NormData) -> None:
        """
        Evaluate Shapiro-Wilk test statistic for normality of residuals.

        The Shapiro-Wilk test assesses whether the z-scores follow a normal distribution.
        A higher W statistic (closer to 1) indicates stronger normality.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values. Must contain 'zscores' variable.
        """
        for responsevar in data.response_var_list:
            resp_predict_data = data.sel({"response_vars": responsevar})
            shapiro_w = self._evaluate_shapiro_w(resp_predict_data)
            data.statistics.loc[{"response_vars": responsevar,
                                 "statistic": "ShapiroW"}] = shapiro_w

    def evaluate_mace(self, data: NormData) -> None:
        """
        Evaluate Mean Absolute Centile Error.
        """
        for responsevar in data.response_var_list:
            resp_predict_data = data.sel({"response_vars": responsevar})
            mace = self._evaluate_mace(resp_predict_data)
            data.statistics.loc[{
                "response_vars": responsevar, "statistic": "MACE"}] = mace

    def evaluate_mape(self, data: NormData) -> None:
        """
        Evaluate Mean Absolute Percentage Error.
        """
        for responsevar in data.response_var_list:
            resp_predict_data = data.sel({"response_vars": responsevar})
            mape = self._evaluate_mape(resp_predict_data)
            data.statistics.loc[{
                "response_vars": responsevar, "statistic": "MAPE"}] = mape

    def evaluate_skew(self, data: NormData) -> None:
        """
        Evaluate the skewness of the z-score distribution.

        Skewness measures asymmetry of the z-score distribution.
        For a well-calibrated normative model the z-scores follow
        a standard normal distribution, so the ideal value is 0.

        Parameters
        ----------
        data : NormData
            Data container with z-scores. Must contain the 'Z'
            variable.
        """
        for responsevar in data.response_var_list:
            # Select data for the current response variable
            resp_predict_data = data.sel(
                {"response_vars": responsevar}
            )
            # Compute skewness and store in the statistics array
            skew = self._evaluate_skew(resp_predict_data)
            data.statistics.loc[{
                "response_vars": responsevar,
                "statistic": "Skew",
            }] = skew

    def evaluate_kurt(self, data: NormData) -> None:
        """
        Evaluate the excess kurtosis of the z-score distribution.

        Excess kurtosis measures the tail heaviness of the z-score
        distribution relative to a normal distribution. For a
        well-calibrated normative model the z-scores follow a
        standard normal distribution, so the ideal value is 0.
        Positive values indicate heavier tails (leptokurtic);
        negative values indicate lighter tails (platykurtic).

        Parameters
        ----------
        data : NormData
            Data container with z-scores. Must contain the 'Z'
            variable.
        """
        for responsevar in data.response_var_list:
            # Select data for the current response variable
            resp_predict_data = data.sel(
                {"response_vars": responsevar}
            )
            # Compute excess kurtosis and store in the statistics array
            kurt = self._evaluate_kurt(resp_predict_data)
            data.statistics.loc[{
                "response_vars": responsevar,
                "statistic": "Kurt",
            }] = kurt

    def _evaluate_rho(self, data: NormData) -> Tuple[float, float]:
        """
        Calculate Spearman's rank correlation coefficient.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values

        Returns
        -------
        float
            Spearman's rank correlation coefficient between actual and predicted values
        """
        y = data["Y"].values
        yhat = data["Yhat"].values
        rho, p_rho = stats.spearmanr(y, yhat)
        return float(rho), float(p_rho)  # type:ignore

    def _evaluate_R2(self, data: NormData) -> float:
        """
        Calculate R2 for model predictions.
        """
        y = data["Y"].values
        yhat = data["Yhat"].values
        r2 = r2_score(y, yhat)
        return float(r2)

    def _evaluate_rmse(self, data: NormData) -> float:
        """
        Calculate Root Mean Square Error.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values

        Returns
        -------
        float
            Root mean square error between actual and predicted values
        """
        y = data["Y"].values
        yhat = data["Yhat"].values
        rmse = np.sqrt(np.mean((y - yhat) ** 2))
        return float(rmse)

    def _evaluate_smse(self, data: NormData) -> float:
        """
        Calculate Standardized Mean Square Error.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values

        Returns
        -------
        float
            Standardized mean square error between actual and predicted values
        """
        y = data["Y"].values
        yhat = data["Yhat"].values

        mse = np.mean((y - yhat) ** 2)
        variance = np.var(y)
        smse = float(mse / variance if variance != 0 else 0)

        return smse

    def _evaluate_expv(self, data: NormData) -> float:
        """
        Calculate Explained Variance score.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values

        Returns
        -------
        float
            Explained variance score between actual and predicted values
        """
        y = data["Y"].values
        yhat = data["Yhat"].values
        return float(explained_variance_score(y, yhat))

    def _evaluate_msll(self, data: NormData) -> float | None:
        """
        Calculate Mean Standardized Log Loss.

        MSLL compares the fitted model's log loss to a baseline Gaussian model.
        Note that both should be computed on the same (scaled) data to ensure 
        proper comparison, as log-likelihoods are scale-dependent.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values.

        Returns
        -------
        float | None
            Mean standardized log loss between actual and predicted values
        """
        # Fitted model mean log loss (negative log-likelihood)
        logp = data["logp"].values
        mll_model = -np.mean(logp)

        # Check that the baseline logp is calculated on the scaled data
        if "baseline_logp" not in data:
            print("Cannot compute MSLL because baseline log probability is "
                  "not computed on scaled data.")
            return None

        # Baseline Gaussian model mean log loss (negative log-likelihood)
        baseline_logp = data["baseline_logp"].values
        mll_baseline = -np.mean(baseline_logp)

        # Compute MSLL (mean standardized log loss)
        msll = mll_model - mll_baseline
        return float(msll)

    def _evaluate_mll(self, data: NormData) -> float:
        """
        Calculate Mean Log Loss.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values. Must
            contain per-observation log-probabilities.

        Returns
        -------
        float
            Mean Log Loss of predictions
        """
        # Emit a deprecation warning using the shared renamed template
        Output.warning(
            Warnings.RENAMED,
            old_name="NLL",
            new_name="MLL",
            category=DeprecationWarning,
        )

        logp = data["logp"].values
        mll = -np.mean(logp)
        return float(mll)

    def _evaluate_bic(self, data: NormData) -> float:
        """
        Calculate Bayesian Information Criterion.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values

        Returns
        -------
        float
            Bayesian Information Criterion value
        """
        n_params = self.n_params()
        y = data["Y"].values
        yhat = data["Yhat"].values

        rss = np.sum((y - yhat) ** 2)
        n = len(y)
        bic = float(n * np.log(rss / n) + n_params *
                    np.log(n))  # Explicitly cast to float
        return bic

    def _evaluate_shapiro_w(self, data: NormData) -> float:
        """
        Calculate Shapiro-Wilk test statistic.

        Parameters
        ----------
        data : NormData
            Data container with z-scores

        Returns
        -------
        float
            Shapiro-Wilk W statistic
        """
        y = data["Z"].values
        shapiro_w, _ = stats.shapiro(y)
        return float(shapiro_w)  # Explicitly cast to float

    def _evaluate_mace(self, data: NormData) -> float:
        """
        Calculate Mean Absolute Centile Error (MACE).

        MACE measures centile calibration by comparing, for each predicted
        centile level, the fraction of subjects whose true value falls below
        that centile curve against the nominal centile value.

        Calibration is computed separately within each
        unique combination of batch effects (e.g.
        site1_male, site1_female, site2_male, ...), and
        the per-combination MACE values are averaged to give
        the final score. This approach prevents
        well-calibrated large groups from masking poor
        calibration at smaller groups.

        This metric is adopted from Zamanzadeh et al. (2026).

        Parameters
        ----------
        data : NormData
            Data container for a single response variable, containing
            Y, centiles, and batch_effects.

        Returns
        -------
        float
            Mean absolute centile error, averaged over all batch groups.
        """
        # True response values
        y = data["Y"].values
        # Nominal centile levels, E.g. [0.05, 0.1, ..., 0.95]
        centile_list = data.centile.values
        # Predicted centile curves for each subject
        centile_data = data.centiles.values

        # Collect one MACE value for each unique batch effect
        batch_mace: list[float] = []

        # Check if there are batch effects
        unique_batch_effects: dict = data.attrs.get("unique_batch_effects", {})
        has_batch = (
            "batch_effects" in data.data_vars
            and len(unique_batch_effects) > 0
        )

        if has_batch:
            # Get batch effects values for each subject
            # eg [['site1', 'M'], ['site1', 'F'], ...]
            be_values = data["batch_effects"].values
            # Get batch effect dimension names
            # eg ['site', 'sex']
            be_dims = list(data.batch_effect_dims.values)

            # Iterate over the non-empty batch combinations provided by
            # shared utility.
            # eg {'site': 'site1', 'sex': 'M'}
            for _, mask in iter_batch_combinations(
                be_values,
                unique_batch_effects,
                be_dims,
            ):

                # Compute the empirical centile for this batch group.
                empirical_centiles = (
                    centile_data[:, mask]
                    >= y[mask]
                ).mean(axis=1)

                # MACE for this combination
                batch_mace.append(
                    float(
                        np.abs(
                            centile_list
                            - empirical_centiles
                        ).mean()
                    )
                )

            # Average MACE across all combinations
            return float(np.mean(batch_mace))

        else:
            # Compute MACE if data have no batch effect
            empirical_centiles = (centile_data >= y).mean(axis=1)
            return float(
                np.abs(centile_list - empirical_centiles).mean()
            )

    def _evaluate_mape(self, data: NormData) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        """
        y = data["Y"].values
        yhat = data["Yhat"].values

        return mean_absolute_percentage_error(y, yhat)

    def _evaluate_skew(self, data: NormData) -> float:
        """
        Calculate the skewness of the z-score distribution.

        Uses the adjusted Fisher-Pearson standardised moment
        coefficient (``bias=False``), which corrects for small-
        sample bias and matches the formula provided by the
        colleague::

            skew = n * m3 / (n-1) / (n-2) / s1**3

        Infinite values in the z-scores are replaced with NaN
        before computation and excluded via ``nan_policy='omit'``.
        Returns NaN if fewer than 3 valid observations are
        available.

        Parameters
        ----------
        data : NormData
            Data container for a single response variable.
            Must contain the 'Z' variable.

        Returns
        -------
        float
            Adjusted sample skewness of the z-scores.
        """
        # Extract raw z-score values as a 1-D float array
        z = data["Z"].values.ravel().astype(np.float64)
        # Replace ±Inf with NaN so they are excluded from the
        # calculation rather than distorting the result
        z[np.isinf(z)] = np.nan
        # Compute adjusted (unbiased) skewness, skipping NaNs
        return float(stats.skew(z, bias=False, nan_policy="omit"))

    def _evaluate_kurt(self, data: NormData) -> float:
        """
        Calculate the excess kurtosis of the z-score distribution.

        Uses the adjusted estimator (``bias=False``) and returns
        **excess** kurtosis (normal distribution = 0), which
        matches the small-sample-corrected formula provided by the
        colleague::

            kurt = (n*(n+1)*m4) / ((n-1)*(n-2)*(n-3)*s1**4)
                   - 3*(n-1)**2 / ((n-2)*(n-3))

        Infinite values in the z-scores are replaced with NaN
        before computation and excluded via ``nan_policy='omit'``.
        Returns NaN if fewer than 4 valid observations are
        available.

        Parameters
        ----------
        data : NormData
            Data container for a single response variable.
            Must contain the 'Z' variable.

        Returns
        -------
        float
            Adjusted excess kurtosis of the z-scores (normal = 0).
        """
        # Extract raw z-score values as a 1-D float array
        z = data["Z"].values.ravel().astype(np.float64)
        # Replace ±Inf with NaN so they are excluded from the
        # calculation rather than distorting the result
        z[np.isinf(z)] = np.nan
        # Compute adjusted excess kurtosis (Fisher's definition,
        # normal distribution = 0), skipping NaNs
        return float(
            stats.kurtosis(z, bias=False, nan_policy="omit")
        )

    def empty_statistic(self) -> xr.DataArray:
        return xr.DataArray(
            np.zeros(len(self.response_vars)),
            dims=("response_vars"),
            coords={"response_vars": self.response_vars},
        )

    def prepare(self, responsevar: str) -> None:
        """Prepare the evaluator for a specific response variable."""
        pass

    def reset(self) -> None:
        """Reset the evaluator state."""
        pass

    def n_params(self) -> int:
        """Return the number of parameters in the model."""
        return 0  # Override in subclasses
