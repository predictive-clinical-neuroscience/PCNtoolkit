"""Tests for data I/O functionality."""

import copy

import numpy as np
import pandas as pd
import pytest

from pcntoolkit.dataio.norm_data import NormData


class TestDataIO:
    """Test data I/O functionality."""

    @pytest.fixture(autouse=True)
    def setup(self, synthetic_data):
        """Setup test environment."""
        self.data = synthetic_data

        # Create NormData object from arrays
        self.norm_data = NormData.from_ndarrays(
            "test_data",
            self.data["covariates"],
            self.data["responses"],
            self.data["batch_effects"],
        )

    def test_norm_data_creation_from_arrays(self):
        """Test creating NormData object from numpy arrays."""
        assert self.norm_data.X.shape == self.data["covariates"].shape
        assert self.norm_data.Y.shape == self.data["responses"].shape
        assert self.norm_data.batch_effects.shape == self.data["batch_effects"].shape
        assert self.norm_data.covariates.shape == (self.data["covariates"].shape[1],)
        assert self.norm_data.batch_effect_dims.shape == (
            self.data["batch_effects"].shape[1],
        )
        assert self.norm_data.coords["observations"].shape == (
            self.data["covariates"].shape[0],
        )
        assert self.norm_data.coords["covariates"].shape == (
            self.data["covariates"].shape[1],
        )
        assert self.norm_data.coords["batch_effect_dims"].shape == (
            self.data["batch_effects"].shape[1],
        )

    def test_norm_data_creation_from_dataframe(self):
        """Test creating NormData object from pandas DataFrame."""
        # Create DataFrame
        df = pd.DataFrame(
            {
                "subject_id": np.arange(self.data["covariates"].shape[0]),
                "age": self.data["covariates"][:, 0],
                "sex": self.data["covariates"][:, 1],
                "response1": self.data["responses"][:, 0],
                "response2": self.data["responses"][:, 1],
                "site": self.data["batch_effects"][:, 0],
                "scanner": self.data["batch_effects"][:, 1],
            }
        )

        # Create NormData object
        norm_data = NormData.from_dataframe(
            "test_data",
            df,
            covariates=["age", "sex"],
            response_vars=["response1", "response2"],
            batch_effects=["site", "scanner"],
            subject_ids="subject_id",
        )

        # Verify structure
        assert norm_data.X.shape == self.data["covariates"].shape
        assert norm_data.Y.shape == self.data["responses"].shape
        assert norm_data.batch_effects.shape == self.data["batch_effects"].shape
        assert norm_data.covariates.shape == (2,)
        assert norm_data.batch_effect_dims.shape == (2,)
        assert norm_data.coords["observations"].shape == (
            self.data["covariates"].shape[0],
        )

    @pytest.mark.parametrize("split_ratio", [(0.5, 0.5), (0.7, 0.3), (0.8, 0.2)])
    def test_train_test_split(self, split_ratio):
        """Test splitting data into train and test sets."""
        splits = self.norm_data.train_test_split(
            splits=split_ratio, split_names=("train", "val")
        )

        # Check basic split properties
        assert len(splits) == 2
        assert splits[0].name == "train"
        assert splits[1].name == "val"

        # Check if total samples in splits equal original samples
        assert (
            splits[0].X.shape[0] + splits[1].X.shape[0]
            == self.data["covariates"].shape[0]
        )

        # Check split sizes
        for i, split in enumerate(splits):
            expected_samples = int(self.data["covariates"].shape[0] * split_ratio[i])
            assert split.X.shape == (expected_samples, self.data["covariates"].shape[1])
            assert split.Y.shape == (expected_samples, self.data["responses"].shape[1])
            assert split.batch_effects.shape == (
                expected_samples,
                self.data["batch_effects"].shape[1],
            )

        # Check stratification
        original_batch_effects = self.norm_data.batch_effects.data
        for col_idx in range(original_batch_effects.shape[1]):
            original_values, original_counts = np.unique(
                original_batch_effects[:, col_idx], return_counts=True
            )
            original_frequencies = original_counts / len(original_batch_effects)

            for split in splits:
                split_values, split_counts = np.unique(
                    split.batch_effects.data[:, col_idx], return_counts=True
                )
                split_frequencies = split_counts / len(split.batch_effects.data)

                # Check if all unique values from original exist in split
                assert np.all(np.isin(original_values, split_values))

                # Check if frequencies are approximately equal (within 10% tolerance)
                for val, orig_freq in zip(original_values, original_frequencies):
                    split_freq = split_frequencies[split_values == val][0]
                    np.testing.assert_allclose(
                        split_freq,
                        orig_freq,
                        atol=0.1,
                        err_msg=f"Frequency mismatch for value {val} in column {col_idx}",
                    )

    def test_chunk(self):
        """Test chunking data into smaller pieces."""
        chunks = list(self.norm_data.chunk(n_chunks=2))
        assert len(chunks) == 2
        for i, chunk in enumerate(chunks):
            assert chunk.response_vars == [f"response_var_{i}"]
            assert chunk.X.shape[0] == self.norm_data.X.shape[0]
            assert chunk.Y.shape[1] == 1

    def test_merge(self):
        """Test merging two NormData objects."""
        datacopy = copy.deepcopy(self.norm_data)
        merged = datacopy.merge(self.norm_data)
        assert merged.X.shape[0] == 2 * self.norm_data.X.shape[0]
        assert merged.Y.shape[0] == 2 * self.norm_data.Y.shape[0]
        assert (
            merged.batch_effects.shape[0] == 2 * self.norm_data.batch_effects.shape[0]
        )

    def test_to_dataframe(self):
        """Test converting NormData to pandas DataFrame."""
        df = self.norm_data.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == self.norm_data.X.shape[0]
        assert all(("X", col) in df.columns for col in self.norm_data.covariates.values)
        assert all(
            ("Y", col) in df.columns for col in self.norm_data.response_vars.values
        )
        assert all(
            ("batch_effects", col) in df.columns
            for col in self.norm_data.batch_effect_dims.values
        )
