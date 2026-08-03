"""
Tests that transferring to several sites at once is equivalent to
transferring to each site separately.

The BLR transfer estimates a mean offset and a variance scaling factor
independently per batch effect group, so the groups never interact. Doing
all sites in one call must therefore give exactly the same coefficients as
doing them one at a time.
"""

import ast

import numpy as np
import pandas as pd
import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.regression_model.blr import BLR


@pytest.fixture(scope="module")
def multisite_data() -> NormData:
    """Build a dataset with one covariate, four sites and two sexes.

    Returns
    -------
    NormData
        Dataset with 4 sites x 2 sexes, each group offset differently so
        that a per-group correction is actually needed.
    """
    rng = np.random.default_rng(42)
    sites = ["site_a", "site_b", "site_c", "site_d"]
    sexes = ["F", "M"]

    n_per_group = 60
    rows = []
    for site_idx, site in enumerate(sites):
        for sex_idx, sex in enumerate(sexes):
            age = rng.uniform(20, 60, n_per_group)
            # Each group gets its own offset and scale, which is exactly what
            # the transfer is supposed to recover.
            offset = 2.0 * site_idx + 0.5 * sex_idx
            scale = 1.0 + 0.2 * site_idx
            y = 0.05 * age + offset + scale * rng.normal(size=n_per_group)
            rows.append(
                pd.DataFrame(
                    {"age": age, "thickness": y, "site": site, "sex": sex}
                )
            )

    df = pd.concat(rows, ignore_index=True)
    df["sub_id"] = df.index.astype(str)

    return NormData.from_dataframe(
        "multisite",
        df,
        covariates=["age"],
        batch_effects=["site", "sex"],
        response_vars=["thickness"],
        subject_ids="sub_id",
    )


def _make_model(save_dir: str) -> NormativeModel:
    """Create a plain BLR normative model.

    Parameters
    ----------
    save_dir : str
        Directory the model writes to.

    Returns
    -------
    NormativeModel
        An unfitted model.
    """
    return NormativeModel(
        template_regression_model=BLR(name="template"),
        savemodel=False,
        evaluate_model=False,
        saveresults=False,
        saveplots=False,
        save_dir=save_dir,
        inscaler="standardize",
        outscaler="standardize",
    )


def _coefficients_by_label(
    model: NormativeModel,
) -> dict[tuple[str, ...], tuple[float, float]]:
    """Return the correction coefficients keyed by batch effect labels.

    The coefficients are stored under the encoded integer ids, and those ids
    are assigned per transfer call. Transferring to one site gives that site
    id 0, so the raw keys of two separate transfers collide. Decoding back to
    labels makes the two routes comparable.

    Parameters
    ----------
    model : NormativeModel
        A transferred model.

    Returns
    -------
    dict[tuple[str, ...], tuple[float, float]]
        Maps a batch effect combination (e.g. ``("F", "site_c")``) to its
        ``(offset, scaling factor)``.
    """
    by_label: dict[tuple[str, ...], tuple[float, float]] = {}
    for response_var in model.response_vars:
        regression_model = model[response_var]
        # Same dimension order as the tuples used to build the keys.
        id_to_label = [
            {idx: label for label, idx in level_map.items()}
            for level_map in regression_model.transfered_be_maps.values()
        ]
        for key, coefficients in regression_model.correction_coefficients.items():
            ids = ast.literal_eval(key)
            labels = tuple(
                id_to_label[dim][value] for dim, value in enumerate(ids)
            )
            by_label[labels] = coefficients
    return by_label


def test_split_keeps_only_its_own_batch_effects(multisite_data: NormData) -> None:
    """A batch effect split must not keep listing the batch effects it dropped."""
    selected, remaining = multisite_data.batch_effects_split(
        {"site": ["site_c", "site_d"]}, names=("selected", "remaining")
    )

    assert sorted(selected.unique_batch_effects["site"]) == ["site_c", "site_d"]
    assert sorted(remaining.unique_batch_effects["site"]) == ["site_a", "site_b"]


def test_transfer_to_multiple_sites_runs(multisite_data: NormData, tmp_path) -> None:
    """Transferring to more than one site at once must not raise."""
    transfer_data, fit_data = multisite_data.batch_effects_split(
        {"site": ["site_c", "site_d"]}, names=("transfer", "fit")
    )

    model = _make_model(str(tmp_path / "base"))
    model.fit(fit_data)

    transferred = model.transfer(transfer_data, save_dir=str(tmp_path / "txfr"))

    assert transferred is not None
    assert sorted(transferred.unique_batch_effects["site"]) == ["site_c", "site_d"]


def test_transfer_all_sites_equals_transfer_each_site(
    multisite_data: NormData, tmp_path
) -> None:
    """Transferring to N sites at once == transferring to each site alone.

    The correction coefficients are estimated per batch effect group with no
    pooling across groups, so both routes must produce identical numbers.
    """
    transfer_sites = ["site_c", "site_d"]
    transfer_data, fit_data = multisite_data.batch_effects_split(
        {"site": transfer_sites}, names=("transfer", "fit")
    )

    model = _make_model(str(tmp_path / "base"))
    model.fit(fit_data)

    # Route 1: both sites in a single transfer call
    together = model.transfer(transfer_data, save_dir=str(tmp_path / "together"))

    # Route 2: one transfer call per site
    separate_coefficients: dict[tuple[str, ...], tuple[float, float]] = {}
    for site in transfer_sites:
        single_site_data, _ = transfer_data.batch_effects_split(
            {"site": [site]}, names=(f"only_{site}", "rest")
        )
        alone = model.transfer(
            single_site_data, save_dir=str(tmp_path / f"alone_{site}")
        )
        separate_coefficients.update(_coefficients_by_label(alone))

    combined_coefficients = _coefficients_by_label(together)

    # Both routes must cover the same groups
    assert set(combined_coefficients) == set(separate_coefficients)

    # ... with the same offset and scaling factor for every group
    for key, (offset, scale) in combined_coefficients.items():
        separate_offset, separate_scale = separate_coefficients[key]
        np.testing.assert_allclose(offset, separate_offset, rtol=1e-10)
        np.testing.assert_allclose(scale, separate_scale, rtol=1e-10)
