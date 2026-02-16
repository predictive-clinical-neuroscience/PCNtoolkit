from typing import List, Tuple

import numpy as np
import xarray as xr
from scipy import stats  # type: ignore
from sklearn.metrics import explained_variance_score, r2_score,mean_absolute_percentage_error

from pcntoolkit.dataio.norm_data import NormData


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
        all_statistics = ["Rho", "Rho_p", "R2", "RMSE", "SMSE", "MSLL", "NLL", "ShapiroW", "MACE", "MAPE", "EXPV"]
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
        if "NLL" in self.statistics:
            self.evaluate_nll(data)
        if "MACE" in self.statistics:
            self.evaluate_mace(data)
        if "MAPE" in self.statistics:
            self.evaluate_mape(data)
        if "EXPV" in self.statistics:
            self.evaluate_expv(data)
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
            np.nan * np.ones((len(data.response_var_list), len(self.statistics))),
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
            data.statistics.loc[{"response_vars": responsevar, "statistic": "Rho"}] = float(rho)
            data.statistics.loc[{"response_vars": responsevar, "statistic": "Rho_p"}] = float(p_rho)

    def evaluate_R2(self, data: NormData) -> None:
        """
        Evaluate R2 for model predictions.
        """
        for responsevar in data.response_var_list:
            resp_predict_data = data.sel(response_vars=responsevar)
            r2 = self._evaluate_R2(resp_predict_data)
            data.statistics.loc[{"response_vars": responsevar, "statistic": "R2"}] = r2

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
            data.statistics.loc[{"response_vars": responsevar, "statistic": "RMSE"}] = rmse

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
            data.statistics.loc[{"response_vars": responsevar, "statistic": "SMSE"}] = smse

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
            data.statistics.loc[{"response_vars": responsevar, "statistic": "EXPV"}] = expv

    def evaluate_msll(self, data: NormData) -> None:
        """
        Evaluate Mean Standardized Log Loss (MSLL) for model predictions.

        MSLL compares the log loss of the model to that of a simple baseline predictor
        that always predicts the mean of the training data.

        MSLL = MLL_model - MLL_null

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values. Must contain 'y', 'Yhat',
            and standard deviation predictions.
        """
        for responsevar in data.response_var_list:
            resp_predict_data = data.sel(response_vars=responsevar)
            msll = self._evaluate_msll(resp_predict_data)
            data.statistics.loc[{"response_vars": responsevar, "statistic": "MSLL"}] = msll

    def evaluate_nll(self, data: NormData) -> None:
        """
        Evaluate Negative Log Likelihood (NLL) for model predictions.

        NLL statistics the probabilistic accuracy of the model's predictions, assuming
        binary classification targets.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values. Must contain 'y' and 'Yhat' variables.
            'y' should contain binary values (0 or 1).
        """
        for responsevar in data.response_var_list:
            resp_predict_data = data.sel(response_vars=responsevar)
            nll = self._evaluate_nll(resp_predict_data)
            data.statistics.loc[{"response_vars": responsevar, "statistic": "NLL"}] = nll

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
            data.statistics.loc[{"response_vars": responsevar, "statistic": "BIC"}] = bic
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
            data.statistics.loc[{"response_vars": responsevar, "statistic": "ShapiroW"}] = shapiro_w

    def evaluate_mace(self, data: NormData) -> None:
        """
        Evaluate Mean Absolute Centile Error.
        """
        for responsevar in data.response_var_list:
            resp_predict_data = data.sel({"response_vars": responsevar})
            mace = self._evaluate_mace(resp_predict_data)
            data.statistics.loc[{"response_vars": responsevar, "statistic": "MACE"}] = mace

    def evaluate_mape(self, data: NormData) -> None:
        """
        Evaluate Mean Absolute Percentage Error.
        """
        for responsevar in data.response_var_list:
            resp_predict_data = data.sel({"response_vars": responsevar})
            mape = self._evaluate_mape(resp_predict_data)
            data.statistics.loc[{"response_vars": responsevar, "statistic": "MAPE"}] = mape

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
        mll_null = -np.mean(baseline_logp)

        # Compute MSLL (mean standardized log loss)
        msll = mll_model - mll_null
        return float(msll)

    def _evaluate_nll(self, data: NormData) -> float:
        """
        Calculate Negative Log Likelihood.

        Parameters
        ----------
        data : NormData
            Data container with predictions and actual values. Assumes binary targets (0 or 1).

        Returns
        -------
        float
            Negative log likelihood of predictions
        """
        logp = data["logp"].values
        nll = -np.mean(logp)
        return float(nll)  # Explicitly cast to float

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
        bic = float(n * np.log(rss / n) + n_params * np.log(n))  # Explicitly cast to float
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
        Calculate Mean Absolute Centile Error.
        """
        y = data["Y"].values
        centile_list = data.centile.values
        centile_data = data.centiles.values
        empirical_centiles = (centile_data >= y).mean(axis=1)
        mace = np.abs(centile_list - empirical_centiles).mean()
        return float(mace)

    def _evaluate_mape(self, data: NormData) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        """
        y = data["Y"].values
        yhat = data["Yhat"].values

        return mean_absolute_percentage_error(y, yhat)


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
