"""A module for plotting functions."""

import copy
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
import scipy.stats as stats
import seaborn as sns  # type: ignore
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.longitudinal_score.longitudinal_score import LongitudinalScore
from pcntoolkit.math_functions.velocity import compute_thrivelines
from pcntoolkit.util.autoscale_plot import autoscale

if TYPE_CHECKING:
    from pcntoolkit.normative_model import NormativeModel

sns.set_theme(style="darkgrid")


def plot_centiles(
    model: "NormativeModel",
    scatter_data: NormData | None = None,
    centiles: list[float] = [0.05, 0.25, 0.5, 0.75, 0.95],
    covariate: str | None = None,
    response_vars: list[str] | None = None,
    scatter_kwargs: dict = {},
    show_figure: bool = True,
    save_dir: str | None = None,
) -> list[Figure]:
    """
    Plot the centiles of the model.

    Parameters
    ----------
    model: NormativeModel
        The model to plot the centiles for.
    scatter_data: NormData
        The data to scatter on top of the centiles.
    centiles: List[float], optional
        The centiles to plot.
    covariate: str, optional
        The covariate to plot on the x-axis.
    response_vars: List[str] | None
        The response vars for which to make the plots. All are plotted if this is None, which is default.        
    scatter_kwargs: dict, optional
        Keyword arguments for the scatter plot.
        May include:
        - color: The color of the scatter points. Hex code or matplotlib color name.
        - alpha: The transparency of the scatter points. Between 0 and 1.
        - s: The size of the scatter points.
        - marker: The marker of the scatter points. Uses matplotlib marker syntax: https://matplotlib.org/stable/api/markers_api.html
        - edgecolor: The edge color of the scatter points. Hex code or
          matplotlib color name.
        - linewidth: The width of the edge of the scatter points.
          0 for no edge.
    show_figure: bool, optional
        If True, call plt.show() after all figures are created.
        Defaults to True.
    save_dir: str | None, optional
        Directory to save the figures. Defaults to None.

    Returns
    -------
    list[Figure]
        One matplotlib Figure per response variable.
    """
    complete_scatter_kwargs: dict = {}
    if scatter_data is not None:
        default_scatter_kwargs = {
            "color": "#f7932f",
            "alpha": min(1, 20 / np.sqrt(len(scatter_data.X))),
            "s": 30,
            "marker": "o",
            "edgecolor": "black",
            "linewidth": 0,
        }
        complete_scatter_kwargs = default_scatter_kwargs | scatter_kwargs


    if covariate is None:
        covariate = model.covariates[0]
        assert isinstance(covariate, str)
    else:
        assert covariate in model.covariates, f"{covariate} is not a valid covariate for the model"
    cov_min = model.covariate_ranges[covariate]["min"]
    cov_max = model.covariate_ranges[covariate]["max"]
    covariate_range = (cov_min, cov_max)

    if response_vars is None:
        response_vars = model.response_vars
    response_vars = list(set(model.response_vars).intersection(set(response_vars)))
    # Select the batch effect that has the most data in the scatter data
    batch_effects = {k: max(v.items(), key=lambda x: x[1])[0] for k, v in model.batch_effect_counts.items()}

    # Create some synthetic data with a single batch effect
    # The plotted covariate is just a linspace
    centile_covariates = np.linspace(covariate_range[0], covariate_range[1], 150)
    centile_df = pd.DataFrame({covariate: centile_covariates})

    # Any other covariates are taken to be the mean of the scatter data, or the midpoint of the covariate range
    for cov in model.covariates:
        if cov != covariate:
            minc = model.covariate_ranges[cov]["min"]
            maxc = model.covariate_ranges[cov]["max"]
            if scatter_data is not None:
                centile_df[cov] = scatter_data.X.sel(covariates=cov).mean().values.item()
            else:
                centile_df[cov] = (minc + maxc) / 2

    # Batch effects are the first ones in the highlighted batch effects
    for be, v in batch_effects.items():
        centile_df[be] = v
    # Assign random values for response vars because they are not needed.
    # They must be > 0 to satisfy later checks that require response_vars > 0.
    for rv in response_vars:
        centile_df[rv] = 1e-6

    centile_data = NormData.from_dataframe(
        "centile",
        dataframe=centile_df,
        covariates=model.covariates,
        response_vars=response_vars,
        batch_effects=list(batch_effects.keys()),
    )  # type:ignore

    if not hasattr(centile_data, "centiles"):
        model.compute_centiles(centile_data, centiles=centiles, recompute=False)

    if not model.has_batch_effect:
        batch_effects = {}

    if scatter_data:
        scatter_data = scatter_data.sel(response_vars = response_vars)

        model.harmonize(scatter_data, reference_batch_effect=batch_effects)

    figs: list[Figure] = []
    for response_var in response_vars:
        # Collect the Figure returned by each per-variable plot call.
        fig = _plot_centiles(
            centile_data=centile_data,
            response_var=response_var,
            covariate=covariate,
            scatter_data=scatter_data,
            scatter_kwargs=complete_scatter_kwargs,
            save_dir=save_dir,
        )
        figs.append(fig)
    # Show all figures at once when requested.
    if show_figure:
        plt.show()
    return figs


def _plot_centiles(
    centile_data: NormData,
    response_var: str,
    covariate: str | None = None,
    scatter_data: NormData | None = None,
    scatter_kwargs: dict = {},
    save_dir: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    sns.set_style("whitegrid")
    # Use the provided axes or create a new figure and axes.
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    filter_dict = {
        "covariates": covariate,
        "response_vars": response_var,
    }

    filtered = centile_data.sel(filter_dict)

    for centile in centile_data.coords["centile"][::-1]:
        d_mean = abs(centile - 0.5)
        if d_mean == 0:
            thickness = 2
        else:
            thickness = 1
        if d_mean <= 0.25:
            style = "-"
        elif d_mean <= 0.475:
            style = "--"
        else:
            style = ":"

        sns.lineplot(
            x=filtered.X,
            y=filtered.centiles.sel(centile=centile),
            color="black",
            linestyle=style,
            linewidth=thickness,
            zorder=2,
            legend="brief",
            ax=ax,
        )

        font = FontProperties()
        font.set_weight("bold")
        ax.text(
            s=centile.item(),
            x=filtered.X[0] - 1,
            y=filtered.centiles.sel(centile=centile)[0],
            color="black",
            horizontalalignment="right",
            verticalalignment="center",
            fontproperties=font,
        )
        ax.text(
            s=centile.item(),
            x=filtered.X[-1] + 1,
            y=filtered.centiles.sel(centile=centile)[-1],
            color="black",
            horizontalalignment="left",
            verticalalignment="center",
            fontproperties=font,
        )

    minx, maxx = ax.get_xlim()
    ax.set_xlim(minx - 0.1 * (maxx - minx), maxx + 0.1 * (maxx - minx))
    if scatter_data:
        scatter_filter = scatter_data.sel(filter_dict)
        df = scatter_filter.to_dataframe()
        data_name = "Y_harmonized"
        columns = [("X", covariate), (data_name, response_var)]
        columns.extend(
            [("batch_effects", be.item()) for be in scatter_data.batch_effect_dims]
        )
        df = df[columns]
        df.columns = [c[1] for c in df.columns]
        sns.scatterplot(
            data=df,
            x=covariate,
            y=response_var,
            ax=ax,
            **scatter_kwargs,
        )

        plotname = (
            f"centiles_{response_var}_{scatter_data.name}_harmonized"
        )
        title = (
            f"Centiles of {response_var}"
            f"\n With harmonized {scatter_data.name} data"
        )
    else:
        plotname = f"centiles_{response_var}"
        title = f"Centiles of {response_var}"

    ax.set_title(title)
    ax.set_xlabel(covariate)
    ax.set_ylabel(response_var)
    # Apply tight layout before saving so it takes effect.
    fig.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(save_dir, f"{plotname}.png"), dpi=300)
        # Close the figure immediately after writing to disk
        plt.close(fig)
    return fig

def plot_centiles_advanced(
    model: "NormativeModel",
    centiles: list[float] | np.ndarray | None = None,
    conditionals: list[float] | np.ndarray | None = None,
    covariate: str | None = None,
    covariate_ranges: dict[str, tuple[float, float]] | None = None,
    response_vars: list[str] | None = None,
    batch_effects: dict[str, list[str]] | None | Literal["all"] = None,
    scatter_data: NormData | None = None,
    harmonize_data: bool = True,
    hue_data: str = "site",
    markers_data: str = "sex",
    show_other_data: bool = False,
    show_figure: bool = True,
    save_dir: str | None = None,
    show_centile_labels: bool = True,
    show_legend: bool = True,
    show_yhat: bool = False,
    plt_kwargs: dict | None = None,
    thrive: LongitudinalScore | None = None,
    thrive_kwargs: dict | None = None,
    **kwargs: Any,
) -> list[Figure]:
    """Generate centile plots for response variables with optional data overlay.

    This function creates visualization of centile curves for all response variables
    in the dataset. It can optionally show the actual data points overlaid on the
    centile curves, with customizable styling based on categorical variables.

    Parameters
    ----------
    model: NormativeModel
        The model to plot the centiles for.
    centiles: List[float] | np.ndarray | None, optional
        The centiles to plot. If None, the default centiles will be used.
    conditionals: List[float] | np.ndarray | None, optional
        A list of x-coordinates for which to plot the conditionals
    covariate: str | None, optional
        The covariate to plot on the x-axis. If None, the first covariate in the model will be used.
    covariate_ranges: tuple[float, float], optional
        The range of the covariate to plot on the x-axis. If None, the range of the covariate that was in the train data will be used.
    response_vars: List[str] | None
        The response vars for which to make the plots. All are plotted if this is None, which is default.
    batch_effects: Dict[str, List[str]] | None | Literal["all"], optional
        The batch effects to plot the centiles for. If None, the batch effect that appears first in alphabetical order will be used.
    scatter_data: NormData | None, optional
        Data to scatter on top of the centiles.
    harmonize_data: bool, optional
        Whether to harmonize the scatter data before plotting. Data will be harmonized to the batch effect for which the centiles were computed.
    hue_data: str, optional
        The column to use for color coding the data. If None, the data will not be color coded.
    markers_data: str, optional
        The column to use for marker styling the data. If None, the data will not be marker styled.
    show_other_data: bool, optional
        Whether to scatter data belonging to groups not in batch_effects.
    show_figure: bool, optional
        If True, call plt.show() after all figures are created.
        Defaults to True.
    save_dir: str | None, optional
        The directory to save the plot to. If None, the plot will not
        be saved.
    show_centile_labels: bool, optional
        Whether to show the centile labels on the plot.
    show_legend: bool, optional
        Whether to show the legend on the plot.
    plt_kwargs: dict, optional
        Additional keyword arguments passed to plt.subplots().
    thrive: LongitudinalScore | None, optional
        Pre-initialized longitudinal score used to obtain the z-score
        correlation matrix via ``get_correlation_matrix()`` (e.g. a fitted
        :class:`~pcntoolkit.longitudinal_score.zgain_score.ZGainScore`).
        When provided, thrivelines are computed and overlaid on the centile
        plot (Z propagation from the score, Y mapping via the normative model).
    thrive_kwargs: dict, optional
        Keyword arguments forwarded to
        :func:`~pcntoolkit.math_functions.velocity.compute_thrivelines`
        (e.g. ``z_anchor_start``, ``z_anchor_end``, ``timepoint_diff``,
        ``covariate_range``, ``z_anchors``). When ``covariate_range`` is omitted,
        it defaults to the min/max covariate in ``thrive.reference_data``, or
        the plotted ``covariate_ranges`` for the thrive covariate. When
        ``z_anchors`` is omitted, it defaults to ``norm.ppf(centiles)`` so
        thrivelines start on the plotted centile curves.
    **kwargs: Any, optional
        Additional keyword arguments for the model.compute_centiles method.

    Returns
    -------
    list[Figure]
        One matplotlib Figure per response variable.
    """
    if covariate is None:
        covariate = model.covariates[0]
        assert isinstance(covariate, str)

    if not covariate_ranges:
        covariate_ranges = {c:defaultdict(lambda: None) for c in model.covariates}
    for c in model.covariates:
        if not covariate_ranges[c]:
            covariate_ranges[c] = (model.covariate_ranges[c]["min"],model.covariate_ranges[c]["max"])
        # cov_min = covariate_ranges[c] or model.covariate_ranges[c]["min"]
        # cov_max = covariate_ranges[c] or model.covariate_ranges[c]["max"]
        # covariate_ranges[c] = (cov_min, cov_max)

    if response_vars is None:
        response_vars = model.response_vars
    response_vars = list(set(model.response_vars).intersection(set(response_vars)))

    if scatter_data:
        # Filter scatter data
        scatter_data = scatter_data.sel(response_vars=response_vars)
        for c in model.covariates:
            cov = scatter_data.X.sel(covariates=c).values
            min, max = covariate_ranges[c]
            idx = np.where((cov >= min) & (cov <= max))[0]
            scatter_data = scatter_data.sel(
                observations=scatter_data.observations[idx]
            )

    if batch_effects == "all":
        if scatter_data:
            batch_effects = scatter_data.unique_batch_effects
        else:
            batch_effects = model.unique_batch_effects
    elif batch_effects is None:
        if scatter_data:
            # Select the first batch effect based on alphabetical order
            batch_effects = {k: [v[0]] for k, v in scatter_data.unique_batch_effects.items()}
        else:
            batch_effects = {k: [v[0]] for k, v in model.unique_batch_effects.items()}

    if plt_kwargs is None:
        plt_kwargs = {}

    # Create some synthetic data with a single batch effect
    # The plotted covariate is just a linspace
    centile_covariates = np.linspace(covariate_ranges[covariate][0], covariate_ranges[covariate][1], 150)
    centile_df = pd.DataFrame({covariate: centile_covariates})

    # Any other covariates are taken to be the mean of the scatter data, or the midpoint of the covariate range
    for cov in model.covariates:
        if cov != covariate:
            minc, maxc = covariate_ranges[cov]
            if scatter_data is not None:
                centile_df[cov] = scatter_data.X.sel(covariates=cov).mean().values.item()
            else:
                centile_df[cov] = (minc + maxc) / 2

    # Batch effects are the first ones in the highlighted batch effects
    for be, v in batch_effects.items():
        centile_df[be] = v[0]
    # Assign random values for response vars because they are not needed.
    # They must be > 0 to satisfy later checks that require response_vars > 0.
    for rv in model.response_vars:
        centile_df[rv] = 1e-6
    centile_data = NormData.from_dataframe(
        "centile",
        dataframe=centile_df,
        covariates=model.covariates,
        response_vars=response_vars,
        batch_effects=list(batch_effects.keys()),
    )  # type: ignore

    conditionals_data: list[NormData] = []
    if conditionals is not None:
        for c in conditionals:
            # Compute the endpoints of the conditional curve (0.01th and 0.99th centile)
            centile = copy.deepcopy(centile_data).isel(observations=[0, 1])
            centile.X.loc[{"covariates": covariate}] = c
            model.compute_centiles(centile, centiles=[0.01, 0.99])

            # Compute the curve in between the endpoints
            conditional_d = copy.deepcopy(centile_data)
            conditional_d.X.loc[{"covariates": covariate}] = c
            for rv in response_vars:
                conditional_d.Y.loc[{"response_vars": rv}] = np.linspace(
                    *(centile.centiles.sel(observations=0, response_vars=rv).values.tolist()), 150
                )
            if not hasattr(conditional_d, "logp"):
                model.compute_logp(conditional_d)
            conditionals_data.append(conditional_d)

    if not hasattr(centile_data, "centiles"):
        model.compute_centiles(centile_data, centiles=centiles, recompute=False, **kwargs)
    if show_yhat and not hasattr(centile_data, "Yhat"):
        model.compute_yhat(centile_data)

    if not model.has_batch_effect:
        batch_effects = {}

    if harmonize_data and scatter_data:
        if model.has_batch_effect:
            reference_batch_effect = {k: v[0] for k, v in batch_effects.items()}
            model.harmonize(scatter_data, reference_batch_effect=reference_batch_effect)
        else:
            model.harmonize(scatter_data)

    # Compute thrivelines once (per response var) and convert them to Y-space.
    thrive_by_region: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if thrive is not None:
        R = _get_score_correlation_matrix(thrive).sel(response_vars=response_vars)
        thrive_covariate = getattr(thrive, "covariate", covariate)
        compute_kwargs = dict(thrive_kwargs or {})
        # Default the covariate range to the reference cohort, else the plot range.
        if "covariate_range" not in compute_kwargs:
            ref_range = _reference_covariate_range(thrive, thrive_covariate)
            if ref_range is not None:
                compute_kwargs["covariate_range"] = ref_range
            elif thrive_covariate in covariate_ranges:
                lo, hi = covariate_ranges[thrive_covariate]
                compute_kwargs["covariate_range"] = (int(round(lo)), int(round(hi)))
        # Default the starting z-scores to the plotted centile curves.
        if (
            "z_anchors" not in compute_kwargs
            and "z_anchor_start" not in compute_kwargs
            and "z_anchor_end" not in compute_kwargs
        ):
            compute_kwargs["z_anchors"] = stats.norm.ppf(
                np.asarray(centile_data.coords["centile"].values, dtype=float)
            )
        thrive_Z, thrive_X = compute_thrivelines(R, **compute_kwargs)
        reference_batch_effects = {k: v[0] for k, v in batch_effects.items()}
        for response_var in response_vars:
            thrive_by_region[response_var] = _thrivelines_to_y(
                model,
                response_var,
                thrive_Z,
                thrive_X,
                thrive_covariate,
                reference_batch_effects,
                centile_data,
            )

    figs: list[Figure] = []
    for response_var in response_vars:
        # Collect the Figure returned by each per-variable plot call.
        fig = _plot_centiles_advanced(
            centile_data=centile_data,
            response_var=response_var,
            covariate=covariate,
            conditionals_data=conditionals_data,
            batch_effects=batch_effects,
            scatter_data=scatter_data,
            harmonize_data=harmonize_data,
            hue_data=hue_data,
            markers_data=markers_data,
            show_other_data=show_other_data,
            save_dir=save_dir,
            show_centile_labels=show_centile_labels,
            show_legend=show_legend,
            show_yhat=show_yhat,
            plt_kwargs=plt_kwargs,
            thrive_xy=thrive_by_region.get(response_var),
        )
        figs.append(fig)
    # Show all figures at once when requested.
    if show_figure:
        plt.show()
    return figs


def _plot_centiles_advanced(
    centile_data: NormData,
    response_var: str,
    covariate: str | None = None,
    conditionals_data: list[NormData] | None = None,
    batch_effects: dict[str, list[str]] | None = None,
    scatter_data: NormData | None = None,
    harmonize_data: bool = True,
    hue_data: str = "site",
    markers_data: str = "sex",
    show_other_data: bool = False,
    save_dir: str | None = None,
    show_centile_labels: bool = True,
    show_legend: bool = True,
    show_yhat: bool = False,
    plt_kwargs: dict | None = None,
    ax: Axes | None = None,
    thrive_xy: tuple[np.ndarray, np.ndarray] | None = None,
) -> Figure:
    sns.set_style("whitegrid")
    # Use provided axes or create a new figure with optional Figure kwargs.
    if ax is None:
        fig, ax = plt.subplots(**(plt_kwargs or {}))
    else:
        fig = ax.get_figure()

    filter_dict = {
        "covariates": covariate,
        "response_vars": response_var,
    }

    filtered = centile_data.sel(filter_dict)

    for centile in centile_data.coords["centile"][::-1]:
        d_mean = abs(centile - 0.5)
        if d_mean == 0:
            thickness = 2
        else:
            thickness = 1
        if d_mean <= 0.25:
            style = "-"

        elif d_mean <= 0.475:
            style = "--"
        else:
            style = ":"

        sns.lineplot(
            x=filtered.X,
            y=filtered.centiles.sel(centile=centile),
            color="black",
            linestyle=style,
            linewidth=thickness,
            zorder=2,
            legend="brief",
            ax=ax,
        )

        font = FontProperties()
        font.set_weight("bold")
        if show_centile_labels:
            ax.text(
                s=centile.item(),
                x=filtered.X[0] - 1,
                y=filtered.centiles.sel(centile=centile)[0],
                color="black",
                horizontalalignment="right",
                verticalalignment="center",
                fontproperties=font,
            )
            ax.text(
                s=centile.item(),
                x=filtered.X[-1] + 1,
                y=filtered.centiles.sel(centile=centile)[-1],
                color="black",
                horizontalalignment="left",
                verticalalignment="center",
                fontproperties=font,
            )
    if show_yhat:
        ax.plot(
            filtered.X,
            filtered.Yhat,
            color="red",
            linestyle="--",
            linewidth=thickness,
            zorder=2,
            label="$\\hat{Y}$",
        )

    minx, maxx = ax.get_xlim()
    ax.set_xlim(minx - 0.1 * (maxx - minx), maxx + 0.1 * (maxx - minx))
    if thrive_xy is not None:
        thrive_x, thrive_y = thrive_xy
        for seg_x, seg_y in zip(thrive_x, thrive_y):
            if np.all(np.isfinite(seg_x)) and np.all(np.isfinite(seg_y)):
                ax.plot(
                    seg_x,
                    seg_y,
                    color="#c0392b",
                    alpha=0.55,
                    linewidth=1.0,
                    zorder=1,
                )
    if scatter_data:
        scatter_filter = scatter_data.sel(filter_dict)
        df = scatter_filter.to_dataframe()
        scatter_data_name = "Y_harmonized" if harmonize_data else "Y"
        columns = [("X", covariate), (scatter_data_name, response_var)]
        columns.extend(
            [("batch_effects", be.item()) for be in scatter_data.batch_effect_dims]
        )
        df = df[columns]
        df.columns = [c[1] for c in df.columns]
        if batch_effects == {}:
            sns.scatterplot(
                df,
                x=covariate,
                y=response_var,
                label=scatter_data.name,
                color="#f7932f",
                alpha=min(1, 20 / np.sqrt(len(scatter_data.X))),
                s=30,
                marker="o",
                edgecolor="black",
                linewidth=0,
                ax=ax,
            )
        else:
            idx = np.full(len(df), True)
            for j in batch_effects:
                idx = np.logical_and(
                    idx,
                    df[j].isin(batch_effects[j]),
                )
            be_df = df[idx]
            scatter = sns.scatterplot(
                data=be_df,
                x=covariate,
                y=response_var,
                hue=hue_data if hue_data in df else None,
                style=markers_data if markers_data in df else None,
                s=50,
                alpha=0.8,
                zorder=1,
                linewidth=0,
                ax=ax,
            )

            if show_other_data:
                non_be_df = df[~idx]
                markers = ["Other data"] * len(non_be_df)
                sns.scatterplot(
                    data=non_be_df,
                    x=covariate,
                    y=response_var,
                    color="#696969",
                    style=markers,
                    markers={"Other data": "s"},
                    linewidth=0,
                    s=10,
                    alpha=0.4,
                    zorder=0,
                    legend=False,
                    ax=ax,
                )

            if show_legend:
                legend = scatter.get_legend()
                if legend:
                    handles = legend.legend_handles
                    labels = [t.get_text() for t in legend.get_texts()]
                    ax.legend(
                        handles,
                        labels,
                        title_fontsize=10,
                    )
            else:
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()

    title = f"Centiles of {response_var}"
    plotname = f"centiles_{response_var}"
    if scatter_data:
        if harmonize_data:
            plotname = (
                f"centiles_{response_var}_{scatter_data.name}_harmonized"
            )
            title = f"{title}\n With harmonized {scatter_data.name} data"
        else:
            plotname = f"centiles_{response_var}_{scatter_data.name}"
            title = f"{title}\n With raw {scatter_data.name} data"

    if conditionals_data:
        for conditional_d in conditionals_data:
            filter_cond = conditional_d.sel(filter_dict)
            x = filter_cond.X
            p = np.exp(filter_cond.logp.values) * 30 + x
            y = filter_cond.Y.values
            ax.plot(
                p,
                y,
                color="#1fbde0",
                linewidth=2,
                zorder=4,
                label="Conditional",
            )
            x = [x[0], x[-1]]
            y = [y[0], y[-1]]
            ax.plot(
                x,
                y,
                color="#1fbde0",
                linewidth=2,
                zorder=4,
                alpha=0.2,
            )

    autoscale(ax=ax)

    ax.set_title(title)
    ax.set_xlabel(covariate)
    ax.set_ylabel(response_var)
    # Apply tight layout before saving so it takes effect.
    fig.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(save_dir, f"{plotname}.png"), dpi=300)
    return fig

def plot_qq(
    data: NormData,
    plt_kwargs: dict | None = None,
    bound: int | float = 0,
    plot_id_line: bool = False,
    hue_data: str | None = None,
    markers_data: str | None = None,
    split_data: str | None = None,
    response_vars: list[str] | None = None,
    seed: int = 42,
    show_figure: bool = True,
    save_dir: str | None = None,
) -> list[Figure]:
    """
    Plot QQ plots for each response variable in the data.

    Parameters
    ----------
    data : NormData
        Data containing the response variables.
    plt_kwargs : dict or None, optional
        Additional keyword arguments for the plot. Defaults to None.
    bound : int or float, optional
        Axis limits for the plot. Defaults to 0.
    plot_id_line : bool, optional
        Whether to plot the identity line. Defaults to False.
    hue_data : str or None, optional
        Column to use for coloring. Defaults to None.
    markers_data : str or None, optional
        Column to use for marker styling. Defaults to None.
    split_data : str or None, optional
        Column to use for splitting data. Defaults to None.
    response_vars: List[str] | None = None,
        The response vars for which to make the plots. All are plotted if this is None, which is default.
    seed : int, optional
        Random seed for reproducibility. Defaults to 42.
    show_figure : bool, optional
        If True, call plt.show() after all figures are created.
        Defaults to True.

    Returns
    -------
    list[Figure]
        One matplotlib Figure per response variable.

    Examples
    --------
    >>> plot_qq(data, plt_kwargs={"figsize": (10, 6)}, bound=3)
    """
    plt_kwargs = plt_kwargs or {}
    if response_vars is None:
        response_vars = data.response_vars.values
    response_vars = list(
        set(data.response_vars.values).intersection(set(response_vars))
    )
    data = data.sel(response_vars=response_vars)
    figs: list[Figure] = []
    for response_var in response_vars:
        # Collect the Figure returned by each per-variable plot call.
        fig = _plot_qq(
            data,
            response_var,
            plt_kwargs,
            bound,
            plot_id_line,
            hue_data,
            markers_data,
            split_data,
            seed,
            save_dir,
        )
        figs.append(fig)
    # Show all figures at once when requested.
    if show_figure:
        plt.show()
    return figs


def _plot_qq(
    data: NormData,
    response_var: str,
    plt_kwargs: dict,
    bound: float = 0,
    plot_id_line: bool = False,
    hue_data: str | None = None,
    markers_data: str | None = None,
    split_data: str | None = None,
    seed: int = 42,
    save_dir: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """
    Plot a QQ plot for a single response variable.

    Parameters
    ----------
    data : NormData
        Data containing the response variable.
    response_var : str
        The response variable to plot.
    plt_kwargs : dict
        Additional keyword arguments for the plot.
    bound : float, optional
        Axis limits for the plot. Not used if 0. Defaults to 0.
    plot_id_line : bool, optional
        Whether to plot the identity line. Defaults to False.
    hue_data : str or None, optional
        Column to use for coloring. Defaults to None.
    markers_data : str or None, optional
        Column to use for marker styling. Defaults to None.
    split_data : str or None, optional
        Column to use for splitting data. Defaults to None.
        All split data will be offset by 1.
    seed : int, optional
        Random seed for reproducibility. Defaults to 42.
    save_dir : str or None, optional
        Directory to save the figure. Defaults to None.
    ax : Axes or None, optional
        Existing axes to draw into. Creates a new figure when None.

    Returns
    -------
    Figure
        The matplotlib Figure containing the QQ plot.

    Examples
    --------
    >>> _plot_qq(data, "response_var", plt_kwargs={"figsize": (10, 6)}, bound=3)
    """
    np.random.seed(seed)
    sns.set_style("whitegrid")
    # Use provided axes or create a new figure and axes.
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    filter_dict = {
        "response_vars": response_var,
    }
    filt = data.sel(filter_dict)

    df: pd.DataFrame = filt.to_dataframe()

    # Create labels for the axes
    tq = "theoretical quantiles"
    rq = f"{response_var} quantiles"

    # Filter columns needed for plotting
    columns = [("Z", response_var)]
    columns.extend([("batch_effects", be.item()) for be in data.batch_effect_dims])
    df = df[columns]
    df.columns = [rq] + [be.item() for be in data.batch_effect_dims]

    # Sort the dataframe by the response variable
    df.sort_values(by=rq, inplace=True)

    # Create a column for the theoretical quantiles
    rand = np.random.randn(df.shape[0])
    rand.sort()
    df[tq] = rand

    if split_data:
        for i, g in enumerate(df.groupby(split_data, sort=False)):
            my_offset = i * 1.0
            my_id = g[1].index
            df.loc[my_id, rq] += i * 1.0
            rand = np.random.randn(g[1].shape[0])
            rand.sort()
            df.loc[my_id, tq] = rand
    alpha = min(1, 20 / np.sqrt(len(df.index)))
    # Plot the QQ-plot
    sns.scatterplot(
        data=df,
        x="theoretical quantiles",
        y=rq,
        hue=hue_data if hue_data in df else None,
        style=markers_data if markers_data in df else None,
        **plt_kwargs,
        linewidth=0,
        alpha=alpha,
        ax=ax,
    )
    if plot_id_line:
        if split_data:
            for i, g in enumerate(df.groupby(split_data, sort=False)):
                my_offset = i * 1.0
                my_id = g[1].index
                ax.plot(
                    [-3, 3],
                    [-3 + my_offset, 3 + my_offset],
                    color="black",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.8,
                    zorder=0,
                )
        else:
            ax.plot(
                [-3, 3],
                [-3, 3],
                color="black",
                linestyle="--",
                linewidth=1,
                alpha=0.8,
                zorder=3,
            )

    if bound != 0:
        ax.axis((-bound, bound, -bound, bound))
    # Apply tight layout before saving so it takes effect.
    fig.tight_layout()
    if save_dir:
        fig.savefig(
            os.path.join(save_dir, f"qq_{response_var}_{data.name}.png"),
            dpi=300,
        )
        # Close the figure immediately after writing to disk
        plt.close(fig)
    return fig


def plot_ridge(
    data: NormData,
    variable: Literal["Z", "Y"],
    split_by: str,
    response_vars: list[str] | None = None,
    show_figure: bool = True,
    save_dir: str | None = None,
    **kwargs: Any,
) -> list[Figure]:
    """
    Plot a ridge plot for each response variable in the data.

    Creates a density plot for the variable split by the split_by variable.

    Each density plot will be on a different row.

    The hue of the density plot will be the split_by variable.

    Parameters
    ----------
    data : NormData
        Data containing the response variable.
    variable : Literal["Z", "Y"]
        The variable to plot on the x-axis. (Z or Y)
    split_by : str
        The variable to split the data by.
    response_vars : list[str] or None, optional
        The response vars for which to make the plots.
        All are plotted if this is None, which is default.
    show_figure : bool, optional
        If True, call plt.show() after all figures are created.
        Defaults to True.
    save_dir : str or None, optional
        The directory to save the plot to. Defaults to None.
    **kwargs : Any, optional
        Additional keyword arguments for the plot.

    Returns
    -------
    list[Figure]
        One matplotlib Figure per response variable.
    """
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    if response_vars is None:
        response_vars = data.response_vars.values
    response_vars = list(
        set(data.response_vars.values).intersection(set(response_vars))
    )
    data = data.sel(response_vars=response_vars)
    figs: list[Figure] = []
    for response_var in response_vars:
        # Collect the Figure returned by each per-variable plot call.
        fig = _plot_ridge(
            data, variable, response_var, split_by, save_dir, **kwargs
        )
        figs.append(fig)
    # Show all figures at once when requested.
    if show_figure:
        plt.show()
    return figs


def _plot_ridge(
    data: NormData,
    variable: str,
    response_var: str,
    split_by: str,
    save_dir: str | None,
    **kwargs: Any,
) -> Figure:
    df = data.to_dataframe()
    # Select only the Z and batch_effects columns
    df = df[[(variable, response_var), ("batch_effects", split_by)]]
    # Join column name levels with an underscore
    df.columns = [df.columns[0][0], df.columns[1][1]]

    # Initialize the FacetGrid object
    palette = kwargs.get(
        "palette",
        sns.cubehelix_palette(
            n_colors=len(df[split_by].unique()), rot=1.5, light=0.7
        ),
    )
    g = sns.FacetGrid(
        df, row=split_by, hue=split_by, aspect=15, height=0.5, palette=palette
    )

    # Draw the densities in a few steps
    g.map(
        sns.kdeplot, variable, bw_adjust=0.5, clip_on=False, fill=True,
        alpha=1, linewidth=1.5,
    )
    g.map(sns.kdeplot, variable, clip_on=False, color="w", lw=2, bw_adjust=0.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x: Any, color: Any, label: str) -> None:
        ax = plt.gca()
        ax.text(
            0, 0.2, label,
            fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes,
        )

    g.map(label, variable)

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-0.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    # Apply tight layout before saving so it takes effect.
    g.figure.tight_layout()
    if save_dir:
        g.figure.savefig(
            os.path.join(
                save_dir,
                f"ridge_{response_var}_{variable}_{split_by}_{data.name}.png",
            ),
            dpi=300,
        )
    return g.figure


def _get_score_correlation_matrix(score: LongitudinalScore) -> xr.DataArray:
    """Return ``score.get_correlation_matrix()`` or raise a clear error."""
    get_matrix = getattr(score, "get_correlation_matrix", None)
    if get_matrix is None or not callable(get_matrix):
        raise TypeError(
            f"{type(score).__name__} does not provide get_correlation_matrix(); "
            "thrivelines require a longitudinal score that estimates "
            "age-to-age z-score correlations (e.g. ZGainScore)."
        )
    return get_matrix()


def _reference_covariate_range(
    thrive: LongitudinalScore,
    covariate: str,
) -> tuple[int, int] | None:
    """Return the integer covariate span of a longitudinal score's reference cohort.

    Used as a default ``covariate_range`` when plotting thrivelines so anchor
    points align with ages (or other integer covariates) present in the
    reference data used to estimate age-to-age correlations.

    Parameters
    ----------
    thrive : LongitudinalScore
        Longitudinal score whose ``reference_data`` defines the cohort span.
    covariate : str
        Covariate to read from ``reference_data.X`` (typically ``"age"``).

    Returns
    -------
    tuple of int or None
        ``(min, max)`` integer covariate values after rounding, or ``None`` if
        ``covariate`` is absent from the reference data or the cohort is empty.
    """
    ref = thrive.reference_data
    covariates = [str(c) for c in ref.covariates.values]
    if covariate not in covariates:
        return None
    ages = np.round(ref.X.sel(covariates=covariate).values.astype(float))
    if ages.size == 0:
        return None
    return int(ages.min()), int(ages.max())


def _encode_batch_effects(
    model: "NormativeModel",
    batch_effects: dict[str, str],
    n_obs: int,
    centile_template: NormData,
) -> xr.DataArray | None:
    """Build a model-encoded batch-effect array for thriveline backward passes.

    Each thriveline segment is evaluated at the reference batch effect used for
    the centile grid. Missing batch-effect keys are filled from
    ``centile_template`` or the model's first allowed level.

    Parameters
    ----------
    model : NormativeModel
        Normative model used to map raw batch labels to encoded values.
    batch_effects : dict of str
        Reference batch labels keyed by batch-effect dimension
        (e.g. ``{"site": "A"}``).
    n_obs : int
        Number of observations (thriveline segments) to replicate batch effects
        for.
    centile_template : NormData
        Synthetic centile grid; used to infer batch levels when a key is absent
        from ``batch_effects``.

    Returns
    -------
    xr.DataArray or None
        Encoded batch effects with dimensions
        ``(observations, batch_effect_dims)``, or ``None`` when the model has
        no batch effects.
    """
    if not model.unique_batch_effects:
        return None
    be_keys = sorted(model.unique_batch_effects.keys())
    be_vals: dict[str, str] = {}
    for k in be_keys:
        if k in batch_effects:
            be_vals[k] = batch_effects[k]
        elif hasattr(centile_template, "batch_effects"):
            be_vals[k] = str(
                centile_template.batch_effects.sel(batch_effect_dims=k)
                .isel(observations=0)
                .values.item()
            )
        else:
            be_vals[k] = model.unique_batch_effects[k][0]
    obs_coord = np.arange(n_obs)
    raw_be = xr.DataArray(
        np.tile([[str(be_vals[k]) for k in be_keys]], (n_obs, 1)),
        dims=("observations", "batch_effect_dims"),
        coords={"observations": obs_coord, "batch_effect_dims": be_keys},
    )
    return model.map_batch_effects(raw_be)


def _thrivelines_to_y(
    model: "NormativeModel",
    response_var: str,
    thrive_Z: xr.DataArray,
    thrive_X: xr.DataArray,
    thrive_covariate: str,
    batch_effects: dict[str, str],
    centile_template: NormData,
) -> tuple[np.ndarray, np.ndarray]:
    """Map propagated Z thrivelines to Y coordinates for plotting.

    For each thriveline segment and offset, covariate values come from
    ``thrive_X``, non-thrive covariates are fixed to the centile-grid means,
    and ``model[response_var].backward`` converts z-scores to response-scale
    ``Y`` at the reference batch effect.

    Parameters
    ----------
    model : NormativeModel
        Normative model providing scalers and the regional ``backward`` map.
    response_var : str
        Response variable to convert.
    thrive_Z : xr.DataArray
        Propagated z-scores from :func:`~pcntoolkit.math_functions.velocity.compute_thrivelines`,
        with dimensions ``(segment, response_vars, offset)``.
    thrive_X : xr.DataArray
        Matching covariate coordinates, same shape as ``thrive_Z``.
    thrive_covariate : str
        Covariate dimension along which thrivelines advance (e.g. ``"age"``).
    batch_effects : dict of str
        Reference batch labels for the centile grid, passed to
        :func:`_encode_batch_effects`.
    centile_template : NormData
        Synthetic centile ``NormData`` used to fix non-thrive covariates and
        infer missing batch levels.

    Returns
    -------
    x_plot, y_plot : tuple of ndarray
        Arrays of shape ``(n_segments, n_offsets)`` with covariate and response
        values ready for matplotlib. Entries are ``NaN`` where propagation or
        lookup failed.
    """
    z_rv = thrive_Z.sel(response_vars=response_var, drop=True)
    x_rv = thrive_X.sel(response_vars=response_var, drop=True)
    n_segments = z_rv.sizes["segment"]
    n_offsets = z_rv.sizes["offset"]

    covariates = list(model.covariates)
    # Hold every covariate except the thrive one at the centile-grid mean.
    fixed_covariates = {
        cov: float(centile_template.X.sel(covariates=cov).mean().values)
        for cov in covariates
        if cov != thrive_covariate
    }
    be_encoded = _encode_batch_effects(model, batch_effects, n_segments, centile_template)
    x_plot = np.full((n_segments, n_offsets), np.nan)
    y_plot = np.full((n_segments, n_offsets), np.nan)

    for o_idx, off in enumerate(z_rv.coords["offset"].values):
        ages = x_rv.sel(offset=off).values.astype(float)
        z_vals = z_rv.sel(offset=off).values.astype(float)
        valid = np.isfinite(ages) & np.isfinite(z_vals)
        if not valid.any():
            continue

        n_valid = int(valid.sum())
        obs_coord = np.arange(n_valid)
        # Build the scaled covariate matrix expected by backward().
        X_scaled = np.zeros((n_valid, len(covariates)))
        for i, cov in enumerate(covariates):
            if cov == thrive_covariate:
                col = ages[valid].reshape(-1, 1)
            else:
                col = np.full((n_valid, 1), fixed_covariates[cov])
            X_scaled[:, i] = model.inscalers[cov].transform(col).ravel()

        X_da = xr.DataArray(
            X_scaled,
            dims=("observations", "covariates"),
            coords={"observations": obs_coord, "covariates": covariates},
        )
        Z_da = xr.DataArray(z_vals[valid], dims=("observations",), coords={"observations": obs_coord})
        be_slice = (
            be_encoded.isel(observations=np.where(valid)[0])
            if be_encoded is not None
            else None
        )
        # Map z-scores to scaled Y, then invert the response scaler.
        y_scaled = model[response_var].backward(X_da, be_slice, Z_da).values
        y_vals = model.outscalers[response_var].inverse_transform(y_scaled.reshape(-1, 1)).ravel()
        x_plot[valid, o_idx] = ages[valid]
        y_plot[valid, o_idx] = y_vals

    return x_plot, y_plot