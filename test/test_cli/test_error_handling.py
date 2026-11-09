import os
import subprocess
import sys

import pytest


class TestCLIErrorHandling:
    """Test CLI error handling and edge cases."""

    @pytest.fixture(autouse=True)
    def setup(self, synthetic_data, temp_output_dir):
        """Setup test environment."""
        self.data = synthetic_data
        self.output_dir = temp_output_dir
        self.save_dir = self.output_dir / "save_dir"
        self.save_dir.mkdir()

    def test_missing_covariates_file(self):
        """Test error handling for missing covariates file."""
        cmd = f"normative -c nonexistent.txt -a blr -r {self.data['resp_path']} save_dir={self.save_dir}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        assert result.returncode != 0
        assert "File not found" in result.stderr

    def test_missing_responses_file(self):
        """Test error handling for missing responses file."""
        cmd = f"normative -c {self.data['cov_path']} -a blr -r nonexistent.txt save_dir={self.save_dir}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        assert result.returncode != 0
        assert "File not found" in result.stderr

    def test_invalid_algorithm(self):
        """Test error handling for invalid algorithm."""
        cmd = f"normative -c {self.data['cov_path']} -a invalid_algo -r {self.data['resp_path']} save_dir={self.save_dir}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        assert result.returncode != 0
        assert "Unknown class" in result.stderr

    def test_invalid_batch_effects_file(self):
        """Test error handling for invalid batch effects file."""
        cmd = f"normative -c {self.data['cov_path']} -a blr -r {self.data['resp_path']} be=nonexistent.txt save_dir={self.save_dir}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        assert result.returncode != 0
        assert "File not found" in result.stderr

    @pytest.mark.skipif(
        sys.platform == "win32", reason="Unix file permissions not available on Windows"
    )
    def test_permission_denied(self):
        """Test error handling for permission denied."""
        # Create a directory with no write permissions
        no_write_dir = self.output_dir / "no_write"
        no_write_dir.mkdir()
        os.chmod(no_write_dir, 0o444)  # Read-only

        cmd = f"normative -c {self.data['cov_path']} -a blr -r {self.data['resp_path']} save_dir={no_write_dir}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        assert result.returncode != 0
        assert "Permission denied" in result.stderr

        # Cleanup
        os.chmod(no_write_dir, 0o777)  # Restore permissions
        no_write_dir.rmdir()
