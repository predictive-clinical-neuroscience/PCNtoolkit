import os
import shutil
import subprocess

import pytest

from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.regression_model.blr import BLR


class TestBasicCLI:
    """Test basic CLI functionality."""

    @pytest.fixture(autouse=True)
    def setup(self, synthetic_data, temp_output_dir):
        """Setup test environment."""
        self.data = synthetic_data
        self.output_dir = temp_output_dir
        self.save_dir = self.output_dir / "save_dir"
        self.save_dir.mkdir()

        # Store original working directory
        self.original_dir = os.getcwd()

        yield

        # Cleanup
        os.chdir(self.original_dir)
        for dir_to_clean in ["temp", "logs"]:
            if os.path.exists(dir_to_clean):
                shutil.rmtree(dir_to_clean)

    def test_simple_blr_cli(self):
        """Test basic CLI command without folds."""
        cmd = f"normative -c {self.data['cov_path']} -a blr -r {self.data['resp_path']} save_dir={self.save_dir}"

        # Run command
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Verify output files
        assert (self.save_dir / "model" / "normative_model.json").exists()
        assert (
            self.save_dir / "plots" / "centiles_response_var_0_fit_data_harmonized.png"
        ).exists()
        assert (self.save_dir / "results" / "centiles_fit_data.csv").exists()

        model = NormativeModel.load(str(self.save_dir))
        assert isinstance(model.template_regression_model, BLR)
        assert not model.template_regression_model.fixed_effect
        assert not model.template_regression_model.fixed_effect_var

        assert not model.template_regression_model.heteroskedastic

    def test_blr_cli_with_folds(self):
        """Test CLI command with cross-validation folds."""
        cmd = f"normative -c {self.data['cov_path']} -a blr -r {self.data['resp_path']} -k 3 save_dir={self.save_dir}"

        # Run command
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Verify output files for each fold
        for fold in range(3):
            fold_dir = self.save_dir / "folds" / f"fold_{fold}"
            assert (fold_dir / "model" / "normative_model.json").exists()
            assert (
                fold_dir
                / "plots"
                / f"centiles_response_var_0_fit_data_fold_{fold}_fit_harmonized.png"
            ).exists()
            assert (
                fold_dir / "results" / f"centiles_fit_data_fold_{fold}_fit.csv"
            ).exists()

    def test_blr_cli_with_batch_effects(self):
        """Test CLI command with batch effect correction."""
        cmd = f"normative -c {self.data['cov_path']} -a blr -r {self.data['resp_path']} -b {self.data['batch_path']} save_dir={self.save_dir}"

        # Run command
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Verify output files
        assert (self.save_dir / "model" / "normative_model.json").exists()
        assert (
            self.save_dir / "plots" / "centiles_response_var_0_fit_data_harmonized.png"
        ).exists()
        assert (self.save_dir / "results" / "centiles_fit_data.csv").exists()

    # def test_blr_cli_with_warp_compose(self):
    #     """Test CLI command with warp compose."""
    #     cmd = f"normative -c {self.data['cov_path']} -a blr -r {self.data['resp_path']} warp=warpcompose[warpsinharcsinh,warpaffine] save_dir={self.save_dir}"

    #     # Run command
    #     result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    #     assert result.returncode == 0, f"Command failed: {result.stderr}"

    #     # Verify output files
    #     assert (self.save_dir / "model" / "normative_model.json").exists()
    #     assert (self.save_dir / "plots" / "centiles_response_var_0_fit_data_harmonized.png").exists()
    #     assert (self.save_dir / "results" / "centiles_fit_data.csv").exists()

    #     # Load model and check if warp compose is used
    #     model = NormativeModel.load(str(self.save_dir))
    #     assert isinstance(model.template_regression_model, BLR)
    #     assert isinstance(model.template_regression_model.warp, WarpCompose)
    #     assert isinstance(model.template_regression_model.warp.warps[0], WarpSinhArcsinh)
    #     assert isinstance(model.template_regression_model.warp.warps[1], WarpAffine)

    # def test_blr_cli_with_warp_compose_and_basis_function(self):
    #     """Test CLI command with warp compose and basis function."""
    #     cmd = f"normative -c {self.data['cov_path']} -a blr -r {self.data['resp_path']} warp=warpcompose[warpboxcox,warpaffine,warpsinharcsinh] basis_function_mean=polynomial basis_function_var=bspline save_dir={self.save_dir}"
    #     print(cmd)
    #     # Run command
    #     result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    #     assert result.returncode == 0, f"Command failed: {result.stderr}"

    #     # Verify output files
    #     assert (self.save_dir / "model" / "normative_model.json").exists()
    #     assert (self.save_dir / "plots" / "centiles_response_var_0_fit_data_harmonized.png").exists()
    #     assert (self.save_dir / "results" / "centiles_fit_data.csv").exists()

    #     # Load model and check if warp compose is used
    #     model = NormativeModel.load(str(self.save_dir))
    #     assert isinstance(model.template_regression_model, BLR)
    #     assert isinstance(model.template_regression_model.warp, WarpCompose)
    #     # Check if basis function is used
    #     assert isinstance(model.template_regression_model.basis_function_mean, PolynomialBasisFunction)
    #     assert model.template_regression_model.basis_function_mean.degree == 3
    #     assert isinstance(model.template_regression_model.basis_function_var, BsplineBasisFunction)
    #     assert model.template_regression_model.basis_function_var.degree == 3

    def test_blr_cli_with_fixed_effects(self):
        """Test CLI command with batch effect correction and warp compose."""
        cmd = f"normative -c {self.data['cov_path']} -a blr -r {self.data['resp_path']} fixed_effect=True heteroskedastic=True intercept_var=True fixed_effect_var=True save_dir={self.save_dir}"
        print(cmd)
        # Run command
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Verify output files
        assert (self.save_dir / "model" / "normative_model.json").exists()
        assert (
            self.save_dir / "plots" / "centiles_response_var_0_fit_data_harmonized.png"
        ).exists()
        assert (self.save_dir / "results" / "centiles_fit_data.csv").exists()

        # Load model and check if batch effect correction is used
        model = NormativeModel.load(str(self.save_dir))
        assert isinstance(model.template_regression_model, BLR)
        assert model.template_regression_model.fixed_effect
        assert model.template_regression_model.fixed_effect_var
        assert model.template_regression_model.heteroskedastic
