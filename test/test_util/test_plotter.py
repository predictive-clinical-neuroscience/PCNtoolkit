"""Tests for the plotter module, focusing on the ax-injection API."""

import matplotlib
# Use non-interactive backend so no display is needed.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.util.plotter import _plot_qq

def _make_norm_data_with_z(
    n: int = 20,
    seed: int = 0,
) -> NormData:
    """Build a tiny NormData that already contains Z-scores.

    Parameters
    ----------
    n : int
        Number of observations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    NormData
        Minimal dataset with one covariate, one response variable,
        one batch-effect dimension, and pre-populated Z scores.
    """
    rng = np.random.default_rng(seed)
    # One covariate (age-like), one response variable, one site column.
    X = rng.uniform(20, 80, (n, 1))
    Y = rng.normal(0, 1, (n, 1))
    batch_effects = rng.choice(["A", "B"], size=(n, 1))

    # Create the base NormData object.
    data = NormData.from_ndarrays(
        name="tiny",
        X=X,
        Y=Y,
        batch_effects=batch_effects,
        subject_ids=np.arange(n),
        attrs={
            "covariates": ["age"],
            "response_vars": ["metric"],
            "batch_effect_dims": ["site"],
        },
    )

    # Inject synthetic Z-scores directly into the dataset.
    z_values = rng.standard_normal((n,))
    data["Z"] = xr.DataArray(
        # Shape must match (observations, response_vars).
        z_values.reshape(n, 1),
        dims=("observations", "response_vars"),
        coords={
            "observations": data.coords["observations"],
            "response_vars": ["metric"],
        },
    )

    return data

def test_001_plot_qq_two_figures_in_subplot():
    """_plot_qq should return the parent Figure when an ax is supplied."""
    # Arrange: one shared figure with two side-by-side axes.
    data = _make_norm_data_with_z()
    fig, (ax_left, ax_right) = plt.subplots(1, 2)

    # Act: draw QQ plots into the two separate axes.
    returned_fig_left = _plot_qq(
        data,
        response_var="metric",
        plt_kwargs={},
        ax=ax_left,
    )
    returned_fig_right = _plot_qq(
        data,
        response_var="metric",
        plt_kwargs={},
        ax=ax_right,
    )

    # Assert: both calls must return the same parent figure.
    assert returned_fig_left is fig
    assert returned_fig_right is fig

    plt.close(fig)


def test_002_plot_qq_should_reflectNewXLabel_when_setAfterReturn():
    """Axes xlabel can be changed after _plot_qq returns."""
    # Arrange
    data = _make_norm_data_with_z()
    fig, (ax_left, _ax_right) = plt.subplots(1, 2)

    # Act: inject the left axes and then rename its x-label.
    _plot_qq(data, response_var="metric", plt_kwargs={}, ax=ax_left)
    ax_left.set_xlabel("custom x label")

    # Assert: the label update is reflected on the axes object.
    assert ax_left.get_xlabel() == "custom x label"

    plt.close(fig)


def test_003_plot_qq_should_reflectNewYLabel_when_setAfterReturn():
    """Axes ylabel can be changed after _plot_qq returns."""
    # Arrange
    data = _make_norm_data_with_z()
    fig, (_ax_left, ax_right) = plt.subplots(1, 2)

    # Act: inject the right axes and then rename its y-label.
    _plot_qq(data, response_var="metric", plt_kwargs={}, ax=ax_right)
    ax_right.set_ylabel("custom y label")

    # Assert
    assert ax_right.get_ylabel() == "custom y label"

    plt.close(fig)


def test_004_plot_qq_should_reflectNewTitle_when_setAfterReturn():
    """Axes title can be changed on both subplot axes after _plot_qq returns."""
    # Arrange
    data = _make_norm_data_with_z()
    fig, (ax_left, ax_right) = plt.subplots(1, 2)

    # Act: draw into both axes, then override titles.
    _plot_qq(data, response_var="metric", plt_kwargs={}, ax=ax_left)
    _plot_qq(data, response_var="metric", plt_kwargs={}, ax=ax_right)
    ax_left.set_title("left panel")
    ax_right.set_title("right panel")

    # Assert: each axes carries its own title.
    assert ax_left.get_title() == "left panel"
    assert ax_right.get_title() == "right panel"

    plt.close(fig)
