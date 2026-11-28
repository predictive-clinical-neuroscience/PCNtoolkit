import numpy as np
import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.regression_model.blr import BLR
from pcntoolkit.util.plotter import plot_centiles, plot_centiles_advanced


class TestNormativeModel:
    """Test normative model functionality."""

    @pytest.fixture(autouse=True)
    def setup(self, synthetic_data, temp_output_dir):
        """Setup test environment."""
        self.data = synthetic_data
        self.output_dir = temp_output_dir
        self.save_dir = self.output_dir / "save_dir"
        self.save_dir.mkdir()

        # Create template regression model
        self.template_model = BLR.from_args(
            "template", {"alg": "blr", "save_dir": str(self.save_dir)}
        )

        # Create normative model
        self.model = NormativeModel(
            template_regression_model=self.template_model,
            savemodel=True,
            evaluate_model=True,
            saveresults=True,
            saveplots=True,
            save_dir=str(self.save_dir),
            inscaler="none",
            outscaler="none",
        )

    def test_model_initialization(self):
        """Test model initialization with valid data."""
        assert self.model.template_regression_model == self.template_model
        assert self.model.save_dir == str(self.save_dir)
        assert self.model.inscaler == "none"
        assert self.model.outscaler == "none"

    def test_model_fit(self):
        """Test model fitting."""
        # Create NormData object
        data = NormData.from_ndarrays(
            name="test_data",
            X=self.data["covariates"],
            Y=self.data["responses"],
            batch_effects=self.data["batch_effects"],
            subject_ids=self.data["subject_ids"],
        )

        # Fit model
        self.model.fit(data)
        assert self.model.is_fitted
        assert len(self.model.regression_models) == 2  # One for each response variable

    def test_model_predict(self):
        """Test model prediction."""
        # Create NormData object
        data = NormData.from_ndarrays(
            name="test_data",
            X=self.data["covariates"],
            Y=self.data["responses"],
            batch_effects=self.data["batch_effects"],
            subject_ids=self.data["subject_ids"],
        )

        self.model.fit(data)

        # Make predictions
        predictions = self.model.predict(data)
        assert "Z" in predictions
        assert predictions.Z.shape == self.data["responses"].shape
        assert not np.isnan(predictions.Z).any()

    def test_model_predict_subset_of_response_vars(self) -> None:
        """Predict and plot data that holds one of the two response variables."""
        data = NormData.from_ndarrays(
            name="test_data",
            X=self.data["covariates"],
            Y=self.data["responses"],
            batch_effects=self.data["batch_effects"],
            subject_ids=self.data["subject_ids"],
        )
        self.model.fit(data)
        subset = data.sel(response_vars=["response_var_0"])

        # saveplots=True: predict ends with plot_centiles
        predictions = self.model.predict(subset)
        assert predictions.Z.shape == (len(self.data["responses"]), 1)

        figs = plot_centiles_advanced(
            self.model,
            scatter_data=subset,
            hue_data="batch_effect_0",
            markers_data="batch_effect_1",
            show_figure=False,
        )
        assert len(figs) == 1

        # Asking for a variable that the scatter data lacks is an error
        with pytest.raises(ValueError, match="none of the response variables"):
            plot_centiles(
                self.model,
                scatter_data=subset,
                response_vars=["response_var_1"],
                show_figure=False,
            )

    def test_model_save_load(self):
        """Test model saving and loading."""
        # Create and fit model
        # Create NormData object
        data = NormData.from_ndarrays(
            name="test_data",
            X=self.data["covariates"],
            Y=self.data["responses"],
            batch_effects=self.data["batch_effects"],
            subject_ids=self.data["subject_ids"],
        )

        self.model.fit(data)

        # Save model
        self.model.save()
        assert (self.save_dir / "model" / "normative_model.json").exists()

        # Load model
        loaded_model = NormativeModel.load(str(self.save_dir))
        assert (
            loaded_model.template_regression_model.__class__
            == self.model.template_regression_model.__class__
        )
        assert loaded_model.is_fitted == self.model.is_fitted

        # Compare predictions
        original_preds = self.model.predict(data)
        loaded_preds = loaded_model.predict(data)
        np.testing.assert_array_almost_equal(original_preds.Z, loaded_preds.Z)

    def test_model_load_after_move(self):
        """A moved model writes its results in the new folder, not the old one."""
        data = NormData.from_ndarrays(
            name="test_data",
            X=self.data["covariates"],
            Y=self.data["responses"],
            batch_effects=self.data["batch_effects"],
            subject_ids=self.data["subject_ids"],
        )
        self.model.fit(data)
        assert not list((self.save_dir / "model").glob("*.tmp"))

        moved_dir = self.output_dir / "moved"
        self.save_dir.rename(moved_dir)
        loaded_model = NormativeModel.load(str(moved_dir))
        assert loaded_model.save_dir == str(moved_dir)

        loaded_model.predict(data)
        assert (moved_dir / "results").exists()
        assert not self.save_dir.exists()

    def test_saveresults_without_evaluation(self) -> None:
        """Results are saved without statistics when evaluate_model=False."""
        data = NormData.from_ndarrays(
            name="test_data",
            X=self.data["covariates"],
            Y=self.data["responses"],
            batch_effects=self.data["batch_effects"],
            subject_ids=self.data["subject_ids"],
        )
        model = NormativeModel(
            template_regression_model=self.template_model,
            savemodel=False,
            evaluate_model=False, # do not evaluate model
            saveresults=True, # save results
            saveplots=False,
            save_dir=str(self.save_dir),
            inscaler="none",
            outscaler="none",
        )

        model.fit_predict(data, data)

        results_dir = self.save_dir / "results"
        for var in ["Z", "centiles", "logp", "Yhat"]:
            assert (results_dir / f"{var}_test_data.csv").exists()
        assert not (results_dir / "statistics_test_data.csv").exists()
        assert "statistics" not in data.data_vars

    def test_model_with_batch_effects(self):
        """Test model with batch effect correction."""
        # Create data with batch effects
        data = NormData.from_ndarrays(
            name="test_data",
            X=self.data["covariates"],
            Y=self.data["responses"],
            batch_effects=self.data["batch_effects"],
            subject_ids=self.data["subject_ids"],
        )

        # Fit and predict
        self.model.fit(data)
        predictions = self.model.predict(data)

        assert "Z" in predictions
        assert predictions.Z.shape == self.data["responses"].shape
        assert not np.isnan(predictions.Z).any()


def test_load_path_must_be_string() -> None:
    """load(None) raises a TypeError saying the path must be a string."""
    with pytest.raises(TypeError, match="Load path must be a string"):
        NormativeModel.load(None)


def test_load_missing_path_raises(tmp_path) -> None:
    """load() on a folder that does not exist raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        NormativeModel.load(str(tmp_path / "missing"))
