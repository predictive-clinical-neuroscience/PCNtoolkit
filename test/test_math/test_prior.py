"""Tests for priors.

Covers the parts of a prior that need no fitted model. Tests that compile a
prior into a PyMC model live in ``test/test_regression_models/test_hbr.py``,
because they need a fitted model to compile against.
"""

import numpy as np
import pytensor.tensor as pt
import pytest

from pcntoolkit.math_functions.prior import BasePrior, make_prior

# --------------------------------------------------------------------------- #
# mappings
# --------------------------------------------------------------------------- #

# Mappings constrain a parameter to the range its likelihood requires: sigma
# must be positive, psi must be a probability, and so on.
MAPPING_INPUT = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])


@pytest.mark.parametrize(
    "mapping, mapping_params, expected",
    [
        ("identity", (0.0, 1.0), [-3.0, -1.0, 0.0, 1.0, 3.0]),
        ("exp", (0.0, 1.0), [0.0498, 0.3679, 1.0, 2.7183, 20.0855]),
        ("softplus", (0.0, 3.0), [0.9398, 1.6209, 2.0794, 2.6209, 3.9398]),
        ("sigmoid", (0.0, 1.0), [0.0474, 0.2689, 0.5, 0.7311, 0.9526]),
        # The first two mapping params shift and scale the input.
        ("sigmoid", (1.0, 2.0), [0.1192, 0.2689, 0.3775, 0.5, 0.7311]),
    ],
)
def test_apply_mapping(mapping, mapping_params, expected):
    prior = make_prior("theta", mapping=mapping, mapping_params=mapping_params)

    mapped = prior.apply_mapping(pt.as_tensor_variable(MAPPING_INPUT)).eval()

    assert np.allclose(mapped, expected, atol=1e-4)

# --------------------------------------------------------------------------- # 
# sigmoid
# --------------------------------------------------------------------------- #

def test_sigmoid_maps_into_unit_interval():
    """The sigmoid mapping exists to keep probabilities (e.g. the psi of a ZINB)
    in (0, 1), so it must hold across the range of plausible sampled values.

    Beyond roughly +-36 the result rounds to exactly 0 or 1 in float64, so this
    checks a range wide enough to cover any realistic sample but short of that.
    """
    prior = make_prior("psi", mapping="sigmoid", mapping_params=(0.0, 1.0))

    mapped = prior.apply_mapping(
        pt.as_tensor_variable(np.linspace(-20, 20, 100))
    ).eval()

    assert np.all(mapped > 0.0)
    assert np.all(mapped < 1.0)


@pytest.mark.parametrize("mapping", ["exp", "softplus"])
def test_positive_mappings_stay_positive(mapping):
    """exp and softplus keep parameters that must be positive, such as sigma.

    As with the sigmoid, very negative inputs round down to exactly 0 in
    float64, so this stops short of that.
    """
    prior = make_prior("sigma", mapping=mapping, mapping_params=(0.0, 1.0))

    mapped = prior.apply_mapping(
        pt.as_tensor_variable(np.linspace(-20, 20, 100))
    ).eval()

    assert np.all(mapped > 0.0)


def test_unknown_mapping_raises():
    prior = make_prior("theta", mapping="not_a_mapping")

    with pytest.raises(ValueError, match="Unknown mapping"):
        prior.apply_mapping(pt.as_tensor_variable(MAPPING_INPUT))


@pytest.mark.parametrize("mapping", ["identity", "exp", "softplus", "sigmoid"])
def test_mapping_survives_serialization(mapping):
    """Models are reloaded from json, so the mapping must be preserved."""
    prior = make_prior("theta", mapping=mapping, mapping_params=(0.5, 2.0))

    loaded = BasePrior.from_dict(prior.to_dict())

    assert loaded.mapping == mapping
    assert loaded.mapping_params == (0.5, 2.0)
