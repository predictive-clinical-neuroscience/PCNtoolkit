import matplotlib
import matplotlib.pyplot as plt

from pcntoolkit.util.plotter import _plot_qq, plot_centiles_advanced
from test.fixtures.blr_model_fixtures import *
from test.fixtures.plotter_fixtures import create_test_data_with_z

# Use non-interactive backend so no display is needed.
matplotlib.use("Agg")


def test_001_plot_qq_two_figures_in_subplot():
    """_plot_qq should return the parent Figure when an ax is supplied."""
    # Arrange: one shared figure with two side-by-side axes.
    data = create_test_data_with_z()
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
    data = create_test_data_with_z()
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
    data = create_test_data_with_z()
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
    data = create_test_data_with_z()
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


def test_005_plot_centiles_advanced_fills_missing_covariate_ranges(
    fitted_norm_blr_model,
):
    """Partial covariate_ranges must be filled from the model (issue #526).

    Ranges given for only one covariate used to raise KeyError on the
    omitted one; they now default to the model's train range.
    """
    # Arrange: two-covariate fitted model, range given for covariate_0 only.
    # Act: plot without supplying a range for covariate_1.
    figures = plot_centiles_advanced(
        fitted_norm_blr_model,
        covariate="covariate_0",
        covariate_ranges={"covariate_0": (0.0, 2.0)},
        show_figure=False,
    )

    # Assert: one figure per response variable, no KeyError raised.
    assert len(figures) == len(fitted_norm_blr_model.response_vars)
    for fig in figures:
        plt.close(fig)


def test_006_plot_centiles_advanced_uses_model_ranges_by_default(
    fitted_norm_blr_model,
):
    """With covariate_ranges=None, all ranges default to the model's."""
    # Act: plot without any ranges at all.
    figures = plot_centiles_advanced(
        fitted_norm_blr_model,
        covariate="covariate_0",
        show_figure=False,
    )

    # Assert: one figure per response variable.
    assert len(figures) == len(fitted_norm_blr_model.response_vars)
    for fig in figures:
        plt.close(fig)
