
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

os.environ["PATH"] = (
    os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
)


@pytest.fixture(autouse=True, scope="session")
def configure_matplotlib():
    """Configure matplotlib for optimal performance during tests."""
    # Use Agg backend for faster rendering
    matplotlib.use("Agg")

    # Disable text rendering optimizations
    plt.rcParams["text.usetex"] = False
    plt.rcParams["text.latex.preamble"] = ""

    # Disable interactive mode
    plt.ioff()

    # Optimize text rendering
    plt.rcParams["text.antialiased"] = True
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]

    # Reduce memory usage
    plt.rcParams["figure.max_open_warning"] = 0

    yield

    # Cleanup
    plt.close("all")


# Basic configuration fixtures
@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(scope="session")
def n_train_subjects():
    return 1500


@pytest.fixture(scope="session")
def n_test_subjects():
    return 1000


@pytest.fixture(scope="session")
def n_covariates():
    return 2


@pytest.fixture(scope="session")
def n_response_vars():
    return 2


@pytest.fixture(scope="session")
def batch_effect_values():
    return [[0, 1], [0, 1, 2]]


# Data generation fixtures
@pytest.fixture(scope="function")
def synthetic_data(
    test_data_dir, n_train_subjects, n_covariates, n_response_vars, batch_effect_values
):
    """Generate synthetic data for testing."""
    # Generate covariates
    X = np.random.rand(n_train_subjects, n_covariates)

    # Generate response variables
    y = np.random.randn(n_train_subjects, n_response_vars)
    noise_coef = np.array([2.3, 1.5])
    slope_coefs = np.array([[1, 0.5, 0.3], [1, 0.5, 0.3]])

    for i in range(n_response_vars):
        y[:, i] = y[:, i] * noise_coef[i] * X[:, 0]
        y[:, i] = (
            y[:, i]
            + slope_coefs[i, 0]
            + X[:, 0] * slope_coefs[i, 1]
            + 0.3 * X[:, 0] ** 2 * slope_coefs[i, 2]
        )

    y = np.square(y)

    # Generate batch effects
    batch_effects = []
    for batch_effect in batch_effect_values:
        batch_effects.append(
            np.random.choice(batch_effect, (n_train_subjects, 1)).astype(int)
        )
    batch_effects = np.concatenate(batch_effects, axis=1)
    subject_ids = np.floor(np.arange(n_train_subjects) / 2).astype(int)

    # Save data to files
    cov_path = test_data_dir / "covariates.txt"
    resp_path = test_data_dir / "responses.txt"
    batch_path = test_data_dir / "batch_effects.txt"
    subject_ids_path = test_data_dir / "subject_ids.txt"

    np.savetxt(cov_path, X)
    np.savetxt(resp_path, y)
    np.savetxt(batch_path, batch_effects)
    np.savetxt(subject_ids_path, subject_ids)

    return {
        "covariates": X,
        "responses": y,
        "batch_effects": batch_effects,
        "subject_ids": subject_ids,
        "cov_path": str(cov_path),
        "resp_path": str(resp_path),
        "batch_path": str(batch_path),
        "subject_ids_path": str(subject_ids_path),
    }


@pytest.fixture(scope="function")
def temp_output_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    yield output_dir
    # Cleanup is handled automatically by tmp_path fixture


@pytest.fixture(scope="function")
def mock_model_config():
    """Provide a mock model configuration for testing."""
    return {
        "algorithm": "blr",
        "n_folds": 3,
        "save_dir": "test_output",
        "log_dir": "test_logs",
    }
