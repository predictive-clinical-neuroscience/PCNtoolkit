"""Covariate selection and saved-model compatibility for composite bases."""

import copy
import json

import numpy as np
import pytest

from pcntoolkit.math_functions.basis_function import (
    BasisFunction,
    BsplineBasisFunction,
    CompositeBasisFunction,
    LinearBasisFunction,
    create_basis_function,
)


@pytest.mark.parametrize("include_untransformed", [False, True])
def test_selection_order_interactions_and_dimension(
    include_untransformed: bool,
) -> None:
    X = np.arange(1.0, 21.0).reshape(5, 4)
    basis = CompositeBasisFunction(
        [LinearBasisFunction(2), LinearBasisFunction(0), LinearBasisFunction(2)],
        interactions=[(0, 1)],
        include_untransformed=include_untransformed,
    )
    basis.fit(X)
    columns = [0, 1, 2, 2, 3] if include_untransformed else [0, 2, 2]
    expected = np.column_stack([X[:, columns], X[:, 2] * X[:, 0]])
    np.testing.assert_allclose(basis.transform(X), expected)
    assert basis.dimension == expected.shape[1]


def test_default_excludes_unselected_covariates() -> None:
    X = np.arange(12.0).reshape(4, 3)
    basis = CompositeBasisFunction([LinearBasisFunction(1)])
    basis.fit(X)
    np.testing.assert_array_equal(basis.transform(X), X[:, [1]])
    changed = X.copy()
    changed[:, [0, 2]] += 1000
    np.testing.assert_array_equal(basis.transform(changed), X[:, [1]])


@pytest.mark.parametrize("include_untransformed", [False, True])
def test_fitted_json_roundtrip(include_untransformed: bool) -> None:
    X = np.column_stack([np.linspace(10, 80, 30), np.arange(30) % 2, np.ones(30)])
    basis = CompositeBasisFunction(
        [BsplineBasisFunction(0), LinearBasisFunction(1)],
        interactions=[(0, 1)],
        include_untransformed=include_untransformed,
    )
    basis.fit(X)
    loaded = BasisFunction.from_dict(
        json.loads(json.dumps(basis.to_dict())), version="1.4.0"
    )
    assert loaded.is_fitted
    assert loaded.include_untransformed is include_untransformed
    assert loaded.dimension == basis.dimension
    np.testing.assert_allclose(loaded.transform(X), basis.transform(X))


@pytest.mark.parametrize("alias", ["Composite", "CompositeBasis"])
@pytest.mark.parametrize("include_untransformed", [False, True])
def test_factory_forwards_selection(alias: str, include_untransformed: bool) -> None:
    basis = create_basis_function(
        alias,
        parts=[LinearBasisFunction(1).to_dict()],
        include_untransformed=include_untransformed,
    )
    X = np.arange(12.0).reshape(4, 3)
    basis.fit(X)
    expected = X if include_untransformed else X[:, [1]]
    np.testing.assert_array_equal(basis.transform(X), expected)


@pytest.mark.parametrize("alias", ["Composite", "CompositeBasis"])
@pytest.mark.parametrize(
    "version, expected", [("1.3.0", True), ("1.4.0", False), (None, True)]
)
def test_legacy_migration(alias: str, version: str | None, expected: bool) -> None:
    payload = {"basis_function": alias, "parts": [LinearBasisFunction(1).to_dict()]}
    basis = BasisFunction.from_dict(payload, version=version)
    assert basis.include_untransformed is expected
    X = np.arange(12.0).reshape(4, 3)
    basis.fit(X)
    np.testing.assert_array_equal(basis.transform(X), X if expected else X[:, [1]])
    assert basis.dimension == (3 if expected else 1)


@pytest.mark.parametrize("alias", ["Composite", "CompositeBasis"])
def test_migration_preserves_explicit_false(alias: str) -> None:
    from pcntoolkit.util.migration import registry

    payload = {"basis_function": alias, "parts": [], "include_untransformed": False}
    assert (
        registry.migrate("BasisFunction", copy.deepcopy(payload), version="1.3.0")
        == payload
    )
    plain = {"basis_function": "LinearBasisFunction", "basis_column": 0}
    assert (
        registry.migrate("BasisFunction", copy.deepcopy(plain), version="1.3.0")
        == plain
    )


@pytest.mark.parametrize("column", [-1, 3, 0.5, True, None])
def test_invalid_column_rejected(column: object) -> None:
    basis = CompositeBasisFunction([LinearBasisFunction(column)])
    with pytest.raises(ValueError, match="basis_column"):
        basis.fit(np.ones((4, 3)))


def test_empty_selection_rejected_and_passthrough_supported() -> None:
    with pytest.raises(ValueError, match="at least one"):
        CompositeBasisFunction([])
    basis = CompositeBasisFunction([], include_untransformed=True)
    X = np.arange(12.0).reshape(4, 3)
    basis.fit(X)
    np.testing.assert_array_equal(basis.transform(X), X)
    assert basis.dimension == 3


def test_passthrough_dimension_requires_fitted_input_width() -> None:
    basis = CompositeBasisFunction([LinearBasisFunction(0)], include_untransformed=True)
    with pytest.raises(ValueError, match="fit"):
        _ = basis.dimension


def test_transform_requires_fit_and_consistent_input_width() -> None:
    basis = CompositeBasisFunction([LinearBasisFunction(0)])
    with pytest.raises(ValueError, match="fit"):
        basis.transform(np.ones((4, 3)))
    basis.fit(np.ones((4, 3)))
    with pytest.raises(ValueError, match="columns"):
        basis.transform(np.ones((4, 2)))


def test_one_dimensional_input_and_refit() -> None:
    basis = CompositeBasisFunction([LinearBasisFunction(0)], include_untransformed=True)
    X = np.arange(4.0)
    basis.fit(X)
    np.testing.assert_array_equal(basis.transform(X), X[:, None])
    assert basis.dimension == 1
    basis.fit(np.column_stack([X, X + 10]))
    assert basis.dimension == 2


@pytest.mark.parametrize("alias", ["Composite", "CompositeBasis"])
def test_fitted_legacy_composite_predicts_without_refitting(alias: str) -> None:
    training = np.column_stack([np.linspace(10, 80, 30) ** 1.2, np.ones(30)])
    basis = CompositeBasisFunction(
        [BsplineBasisFunction(0, knot_method="quantile", include_linear=True)],
        include_untransformed=True,
    )
    basis.fit(training)
    prediction = training[:5].copy()
    prediction[:, 0] += 5
    expected = basis.transform(prediction)
    payload = basis.to_dict()
    payload["basis_function"] = alias
    for key in ("is_fitted", "n_features_in", "include_untransformed"):
        payload.pop(key)
    # Old B-splines implicitly included their linear term.
    payload["parts"][0].pop("include_linear")
    restored = BasisFunction.from_dict(payload, version="1.3.0")
    np.testing.assert_allclose(restored.transform(prediction), expected)
    assert restored.is_fitted
    assert restored.dimension == expected.shape[1]
    np.testing.assert_array_equal(restored.parts[0].knots, basis.parts[0].knots)
