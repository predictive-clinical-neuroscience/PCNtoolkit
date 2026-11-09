
import pytest

from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.regression_model.blr import BLR
from pcntoolkit.regression_model.hbr import HBR


class TestNormFactory:
    """Test normative model factory functionality."""

    @pytest.fixture(autouse=True)
    def setup(self, temp_output_dir):
        """Setup test environment."""
        self.output_dir = temp_output_dir
        self.save_dir = self.output_dir / "save_dir"
        self.save_dir.mkdir()

    def test_create_blr_model(self):
        """Test creating a BLR model."""
        model = NormativeModel.from_args(
            alg="blr",
            save_dir=str(self.save_dir),
            savemodel=True,
            evaluate_model=True,
            saveresults=True,
            saveplots=True,
            inscaler="none",
            outscaler="none",
        )

        assert isinstance(model.template_regression_model, BLR)
        assert model.save_dir == str(self.save_dir)
        assert model.inscaler == "none"
        assert model.outscaler == "none"

    def test_create_hbr_model(self):
        """Test creating an HBR model."""
        model = NormativeModel.from_args(
            alg="hbr",
            save_dir=str(self.save_dir),
            savemodel=True,
            evaluate_model=True,
            saveresults=True,
            saveplots=True,
            inscaler="none",
            outscaler="none",
        )

        assert isinstance(model.template_regression_model, HBR)
        assert model.save_dir == str(self.save_dir)
        assert model.inscaler == "none"
        assert model.outscaler == "none"

    def test_invalid_algorithm(self):
        """Test creating a model with invalid algorithm."""
        with pytest.raises(ValueError):
            NormativeModel.from_args(alg="invalid", save_dir=str(self.save_dir))

    def test_missing_algorithm(self):
        """Test creating a model without specifying algorithm."""
        with pytest.raises(AssertionError, match="Algorithm must be specified"):
            NormativeModel.from_args(save_dir=str(self.save_dir))
