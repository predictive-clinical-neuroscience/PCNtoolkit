"""Tests for utility functions."""

import os
import shutil

import pytest

from pcntoolkit.normative import NormativeModel
from pcntoolkit.util.runner import Runner
from test.fixtures.blr_model_fixtures import *
from test.fixtures.norm_data_fixtures import *
from test.fixtures.test_model_fixtures import *


class TestRunner:
    """Test the Runner utility class."""

    @pytest.fixture(autouse=True)
    def setup(self, synthetic_data, temp_output_dir, norm_data_from_arrays, test_model):
        """Setup test environment."""
        self.data = synthetic_data
        self.output_dir = temp_output_dir
        self.save_dir = self.output_dir / "save_dir"
        self.save_dir.mkdir()
        self.norm_data = norm_data_from_arrays

        # Create model with BLR template
        self.model = NormativeModel(template_regression_model=test_model, save_dir=str(self.save_dir))

    def cleanup(self, model, runner):
        """Clean up test files."""
        if os.path.exists(model.save_dir):
            shutil.rmtree(model.save_dir)
        if os.path.exists(runner.log_dir):
            shutil.rmtree(runner.log_dir)
        if os.path.exists(runner.temp_dir):
            shutil.rmtree(runner.temp_dir)

    def test_fit(self):
        """Test model fitting."""
        runner = Runner(cross_validate=False, parallelize=False)
        runner.fit(self.model, self.norm_data, observe=True)
        assert self.model.is_fitted
        assert os.path.exists(os.path.join(self.model.save_dir, "model", "normative_model.json"))
        self.cleanup(self.model, runner)

    def test_fit_kfold(self):
        """Test k-fold cross-validation fitting."""
        runner = Runner(cross_validate=True, cv_folds=2, parallelize=False)
        runner.fit(self.model, self.norm_data, observe=True)
        assert self.model.is_fitted
        assert os.path.exists(os.path.join(self.model.save_dir, "folds", "fold_0", "model", "normative_model.json"))
        assert os.path.exists(os.path.join(self.model.save_dir, "folds", "fold_1", "model", "normative_model.json"))
        self.cleanup(self.model, runner)

    def test_predict(self):
        """Test model prediction."""
        # First fit the model
        runner = Runner(cross_validate=False, parallelize=False)
        runner.fit(self.model, self.norm_data, observe=True)

        # Then predict
        runner.predict(self.model, self.norm_data, observe=True)
        assert self.model.is_fitted
        assert os.path.exists(os.path.join(self.model.save_dir, "model", "normative_model.json"))
        assert os.path.exists(os.path.join(self.model.save_dir, "results"))
        assert os.path.exists(
            os.path.join(
                self.model.save_dir,
                "plots",
                f"centiles_{self.norm_data.response_vars.values[0]}_{self.norm_data.name}_harmonized.png",
            )
        )
        self.cleanup(self.model, runner)

    def test_predict_kfold_error(self):
        """Test that predict raises error for k-fold without fitting."""
        runner = Runner(cross_validate=True, cv_folds=2, parallelize=False)
        with pytest.raises(ValueError):
            runner.predict(self.model, self.norm_data, observe=True)

    def test_fit_predict(self):
        """Test fit and predict in one step."""
        train, test = self.norm_data.train_test_split(splits=[0.2, 0.8])
        runner = Runner(cross_validate=False, parallelize=False)
        runner.fit_predict(self.model, train, test, observe=True)
        assert self.model.is_fitted
        assert os.path.exists(os.path.join(self.model.save_dir, "model", "normative_model.json"))
        assert os.path.exists(os.path.join(self.model.save_dir, "results"))
        assert os.path.exists(
            os.path.join(
                self.model.save_dir,
                "plots",
                f"centiles_{test.response_vars.values[0]}_{test.name}_harmonized.png",
            )
        )
        self.cleanup(self.model, runner)

    def test_fit_predict_kfold(self):
        """Test fit and predict with k-fold cross-validation."""
        train, test = self.norm_data.train_test_split(splits=[0.2, 0.8])
        runner = Runner(cross_validate=True, cv_folds=2, parallelize=False)
        runner.fit_predict(self.model, train, test, observe=True)
        assert self.model.is_fitted
        assert os.path.exists(os.path.join(self.model.save_dir, "folds", "fold_0", "model", "normative_model.json"))
        assert os.path.exists(os.path.join(self.model.save_dir, "folds", "fold_1", "model", "normative_model.json"))
        assert os.path.exists(os.path.join(self.model.save_dir, "folds", "fold_0", "results"))
        assert os.path.exists(os.path.join(self.model.save_dir, "folds", "fold_1", "results"))
        assert os.path.exists(
            os.path.join(
                self.model.save_dir,
                "folds",
                "fold_0",
                "plots",
                f"centiles_{train.response_vars.values[0]}_{train.name}_fold_0_predict_harmonized.png",
            )
        )
        assert os.path.exists(
            os.path.join(
                self.model.save_dir,
                "folds",
                "fold_1",
                "plots",
                f"centiles_{train.response_vars.values[0]}_{train.name}_fold_1_train_harmonized.png",
            )
        )
        self.cleanup(self.model, runner)

    def test_extend(self):
        """Test model extension."""
        # First fit the model
        runner = Runner(cross_validate=False, parallelize=False)
        runner.fit(self.model, self.norm_data, observe=True)

        # Then extend
        extend_dir = os.path.join(self.model.save_dir, "extend")
        if os.path.exists(extend_dir):
            shutil.rmtree(extend_dir)
        os.makedirs(extend_dir, exist_ok=True)

        extended_model = runner.extend(self.model, self.norm_data, extend_dir, observe=True)
        assert isinstance(extended_model, NormativeModel)
        assert extended_model.is_fitted
        assert os.path.exists(os.path.join(extended_model.save_dir, "model", "normative_model.json"))
        assert os.path.exists(os.path.join(extended_model.save_dir, "results"))
        assert os.path.exists(
            os.path.join(
                extended_model.save_dir,
                "plots",
                f"centiles_{self.norm_data.response_vars.values[0]}_from_arrays_+_synthesized_harmonized.png",
            )
        )
        self.cleanup(extended_model, runner)

    def test_extend_predict(self):
        """Test model extension and prediction."""
        # First fit the model
        runner = Runner(cross_validate=False, parallelize=False)
        runner.fit(self.model, self.norm_data, observe=True)

        # Then extend and predict
        train, test = self.norm_data.train_test_split(splits=[0.2, 0.8])
        extend_dir = os.path.join(self.model.save_dir, "extend_predict")
        if os.path.exists(extend_dir):
            shutil.rmtree(extend_dir)
        os.makedirs(extend_dir, exist_ok=True)

        extended_model = runner.extend_predict(self.model, train, test, extend_dir, observe=True)
        assert isinstance(extended_model, NormativeModel)
        assert extended_model.is_fitted
        assert os.path.exists(os.path.join(extended_model.save_dir, "model", "normative_model.json"))
        assert os.path.exists(os.path.join(extended_model.save_dir, "results"))
        assert os.path.exists(
            os.path.join(
                extended_model.save_dir,
                "plots",
                f"centiles_{test.response_vars.values[0]}_{test.name}_harmonized.png",
            )
        )
        self.cleanup(extended_model, runner)
