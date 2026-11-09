"""Smoke test for an installed PCNtoolkit.

Run this script from a directory outside the source tree, so that Python
imports the installed package and not the local checkout. It fits a BLR and
an HBR model on a small synthetic dataset, then saves, loads and predicts
again with the BLR model. The goal is to find install and platform problems
(for example a missing compiler on Windows), not to check model accuracy.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import pcntoolkit
from pcntoolkit import BLR, HBR, NormativeModel, NormData


def make_data(n_obs: int = 200, seed: int = 42) -> tuple[NormData, NormData]:
    """Make a small synthetic train and test dataset.

    Parameters
    ----------
    n_obs : int
        Number of observations in each split.
    seed : int
        Seed for the random number generator.

    Returns
    -------
    tuple[NormData, NormData]
        The train and the test data.
    """
    rng = np.random.default_rng(seed)
    age = rng.uniform(10, 80, size=2 * n_obs)
    site = rng.choice(["site_a", "site_b"], size=2 * n_obs)
    sex = rng.choice(["F", "M"], size=2 * n_obs)
    df = pd.DataFrame(
        {
            "age": age,
            "site": site,
            "sex": sex,
            "thickness": 3.0 - 0.01 * age + rng.normal(0, 0.1, size=2 * n_obs),
            "volume": 100.0 + 0.2 * age + rng.normal(0, 5.0, size=2 * n_obs),
        }
    )
    kwargs = {
        "covariates": ["age"],
        "batch_effects": ["site", "sex"],
        "response_vars": ["thickness", "volume"],
    }
    train = NormData.from_dataframe("train", df.iloc[:n_obs], **kwargs)
    test = NormData.from_dataframe("test", df.iloc[n_obs:], **kwargs)
    return train, test


def check_z(data: NormData, label: str) -> None:
    """Check that predictions have finite Z-scores of the correct shape.

    Parameters
    ----------
    data : NormData
        The data after ``predict``.
    label : str
        Name of the check, used in the output.

    Raises
    ------
    AssertionError
        If the Z-scores are missing, have the wrong shape or are not finite.
    """
    assert "Z" in data, f"{label}: no Z-scores in the predictions"
    z = data["Z"].values
    assert z.shape == data["Y"].shape, f"{label}: Z shape {z.shape} != Y shape"
    assert np.isfinite(z).all(), f"{label}: Z-scores are not all finite"
    print(f"[smoke] {label}: OK, Z shape {z.shape}")


def run_blr(save_dir: Path) -> None:
    """Fit, save, load and predict with a BLR model.

    Parameters
    ----------
    save_dir : Path
        Folder for the model and its results.
    """
    train, test = make_data()
    model = NormativeModel(BLR(), save_dir=str(save_dir), saveplots=False)
    check_z(model.fit_predict(train, test), "BLR fit_predict")

    loaded = NormativeModel.load(str(save_dir))
    _, test_again = make_data()
    check_z(loaded.predict(test_again), "BLR load and predict")


def run_hbr(save_dir: Path) -> None:
    """Fit and predict with a very short HBR sampling run.

    Parameters
    ----------
    save_dir : Path
        Folder for the model and its results.
    """
    train, test = make_data()
    # Few draws: this only checks that sampling works. Two chains on two
    # cores test parallel sampling, which users get by default.
    hbr = HBR(draws=20, tune=20, chains=2, cores=2, progressbar=False)
    model = NormativeModel(hbr, save_dir=str(save_dir), saveplots=False)
    check_z(model.fit_predict(train, test), "HBR fit_predict")


def main() -> int:
    """Run all smoke tests.

    Returns
    -------
    int
        Exit code, 0 on success.
    """
    print(f"[smoke] pcntoolkit {pcntoolkit.__version__} from {pcntoolkit.__file__}")
    print(f"[smoke] python {sys.version}")
    with tempfile.TemporaryDirectory() as tmp:
        run_blr(Path(tmp) / "blr")
        run_hbr(Path(tmp) / "hbr")
    print("[smoke] all checks passed")
    return 0


# macOS and Windows start worker processes with spawn, which imports this
# file again; the guard stops each worker from rerunning the tests.
if __name__ == "__main__":
    sys.exit(main())
