
import numpy as np
import xarray as xr

from pcntoolkit.regression_model.regression_model import RegressionModel


class TestModel(RegressionModel):
    """
    Test model for regression model testing.
    """

    def __init__(self, name: str, success_ratio: float = 1.0):
        """
        Initialize the test model.

        Args:
            name: The name of the model.
            success_ratio: The ratio of successful fits.
        """
        super().__init__(name)
        self.success_ratio = success_ratio

    def fit(
        self,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ) -> None:
        success = np.random.rand() < self.success_ratio
        if success:
            self.is_fitted = True
        else:
            raise ValueError("Failed to fit model")

    def forward(
        self, X: xr.DataArray, be: xr.DataArray, Y: xr.DataArray
    ) -> xr.DataArray:
        return (Y.copy() - X.sel(covariates=X.covariates.values[0]).values) / X.sel(
            covariates=X.covariates.values[0]
        ).values

    def backward(
        self, X: xr.DataArray, be: xr.DataArray, Z: xr.DataArray
    ) -> xr.DataArray:
        return (
            Z.copy() * X.sel(covariates=X.covariates.values[0]).values
            + X.sel(covariates=X.covariates.values[0]).values
        )

    def elemwise_logp(
        self, X: xr.DataArray, be: xr.DataArray, Y: xr.DataArray
    ) -> xr.DataArray:
        logp = np.random.randn(*Y.shape)
        return xr.DataArray(logp, coords=Y.coords, dims=Y.dims, attrs=Y.attrs)

    def transfer(
        self,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ) -> RegressionModel:
        return TestModel(self.name, self.success_ratio)

    def to_dict(self, path: str | None = None) -> dict:
        mydict = self.regmodel_dict
        mydict["name"] = self.name
        mydict["success_ratio"] = self.success_ratio
        return mydict

    @classmethod
    def from_dict(cls, my_dict: dict, path: str) -> RegressionModel:
        return TestModel(my_dict["name"], my_dict["success_ratio"])

    @classmethod
    def from_args(cls, name: str, args: dict) -> RegressionModel:
        return TestModel(name, args["success_ratio"])

    @property
    def has_batch_effect(self) -> bool:
        return False
