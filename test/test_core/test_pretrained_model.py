"""
Pretrained models in test/resources/pretrained_* were produced by an actual training
run under PCNtoolkit 1.1.2 with a single response variable. They test
pcntoolkit/util/migration.py which was introduced in v1.1.2.
"""

from pathlib import Path

import pytest

from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.regression_model.blr import BLR
from pcntoolkit.regression_model.hbr import HBR

RESOURCES: Path = Path(__file__).parents[1] / "resources"
PRETRAINED_BLR: Path = RESOURCES / "pretrained_blr"
PRETRAINED_HBR: Path = RESOURCES / "pretrained_hbr"

# Structure of the committed pretrained model, as generated with PCNtoolkit 1.1.2.
RESPONSE_VAR: str = "response_var_0"
COVARIATE: str = "covariate_0"
BATCH_EFFECTS: dict[str, list[str]] = {"site": ["0", "1"]}


def _load_pretrained_model(path: Path) -> NormativeModel:
    """Load a committed pre-trained model, failing clearly if it is missing.

    Parameters
    ----------
    path : Path
        Directory containing the ``model/`` subtree.

    Returns
    -------
    NormativeModel
        The loaded model.

    Raises
    ------
    AssertionError
        If the pretrained model is not on disk.
    """
    manifest = path / "model" / "normative_model.json"
    assert manifest.exists(), (
        f"Missing pre-trained pretrained model at {manifest}. "
    )
    return NormativeModel.load(str(path))


@pytest.fixture(scope="module")
def blr_model() -> NormativeModel:
    """Pre-trained BLR model saved with PCNtoolkit 1.1.2."""
    return _load_pretrained_model(PRETRAINED_BLR)


@pytest.fixture(scope="module")
def hbr_model() -> NormativeModel:
    """Pre-trained HBR model saved with PCNtoolkit 1.1.2."""
    return _load_pretrained_model(PRETRAINED_HBR)


# ---------------------------------------------------------------------------
# BLR
# ---------------------------------------------------------------------------


def test_001_load_should_returnFittedModel_when_givenPretrainedBLR(
    blr_model: NormativeModel,
) -> None:
    """
    Arrange: a BLR model saved with PCNtoolkit 1.1.2.
    Act: load it with the installed version.
    Assert: it loads and reports itself as fitted.
    """
    assert isinstance(blr_model, NormativeModel)
    assert blr_model.is_fitted


def test_002_load_should_restoreStructure_when_givenPretrainedBLR(
    blr_model: NormativeModel,
) -> None:
    """
    Arrange: a BLR model saved with PCNtoolkit 1.1.2.
    Act: inspect the restored metadata.
    Assert: covariates and batch effects survived the round trip.
    """
    assert blr_model.covariates == [COVARIATE]
    assert blr_model.unique_batch_effects == BATCH_EFFECTS
    assert blr_model.inscaler == "standardize"
    assert blr_model.outscaler == "standardize"


def test_003_load_should_restoreRegressionModel_when_givenPretrainedBLR(
    blr_model: NormativeModel,
) -> None:
    """
    Arrange: a BLR model saved with PCNtoolkit 1.1.2.
    Act: inspect the restored regression models.
    Assert: the BLR type was dispatched correctly and is fitted.

    It checks you got a real fitted BLR back, not an empty placeholder.

    It checks that:
    - the response variable is actually there
    - it's a BLR (not some other regression type)
    - it's fitted: it has the trained coefficients, not blank ones
    - the template is a BLR too
    """
    assert RESPONSE_VAR in blr_model.regression_models
    regression_model = blr_model.regression_models[RESPONSE_VAR]
    assert isinstance(regression_model, BLR)
    assert regression_model.is_fitted
    assert isinstance(blr_model.template_regression_model, BLR)


# ---------------------------------------------------------------------------
# HBR
# ---------------------------------------------------------------------------


def test_004_load_should_returnFittedModel_when_givenPretrainedHBR(
    hbr_model: NormativeModel,
) -> None:
    """
    Arrange: an HBR model saved with PCNtoolkit 1.1.2.
    Act: load it with the installed version.
    Assert: it loads and reports itself as fitted.

    Regression test: HBR.from_dict reads inference_method, vi_iterations and
    vi_draws as required keys. Models saved before variational inference was
    added do not have them, so without a registered HBR migration this raises
    KeyError.
    """
    assert isinstance(hbr_model, NormativeModel)
    assert hbr_model.is_fitted


def test_005_load_should_restoreRegressionModel_when_givenPretrainedHBR(
    hbr_model: NormativeModel,
) -> None:
    """
    Arrange: an HBR model saved with PCNtoolkit 1.1.2.
    Act: inspect the restored regression models.
    Assert: the HBR type was dispatched correctly and is fitted.
    """
    assert RESPONSE_VAR in hbr_model.regression_models
    regression_model = hbr_model.regression_models[RESPONSE_VAR]
    assert isinstance(regression_model, HBR)
    assert regression_model.is_fitted


def test_006_load_should_deserializeIdata_when_givenPretrainedHBR(
    hbr_model: NormativeModel,
) -> None:
    """
    Arrange: an HBR model saved with PCNtoolkit 1.1.2, whose posterior samples
        live in an accompanying idata.nc.
    Act: inspect the inference data on the restored regression model.
    Assert: the NetCDF file deserialized into a populated posterior group.

    This is the only test covering the arviz/xarray read path, where an ArviZ or
    xarray major-version change would surface.
    """
    regression_model = hbr_model.regression_models[RESPONSE_VAR]
    assert regression_model.idata is not None
    # idata is an xr.DataTree, so groups are children, not arviz groups().
    assert "posterior" in regression_model.idata.children
    assert regression_model.idata["posterior"].dataset.sizes["draw"] > 0


def test_007_load_should_applyVariationalInferenceMigration_when_givenPretrainedHBR(
    hbr_model: NormativeModel,
) -> None:
    """
    Arrange: an HBR model saved with PCNtoolkit 1.1.2, before variational
        inference existed.
    Act: load it with the installed version.
    Assert: the variational inference fields hold the MCMC defaults.

    Tests specifically that _migrate_hbr_1_3_1 correctly sets the inference_method to 
    "mcmc" for a model saved with pcntoolkit v1.1.2.
    """
    regression_model = hbr_model.regression_models[RESPONSE_VAR]
    assert regression_model.inference_method == "mcmc"
