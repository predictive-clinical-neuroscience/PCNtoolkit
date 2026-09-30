"""
The saved model JSON file can change structure across PCNtoolkit versions as
new features are added. This module is one central place to register
and apply migrations required to load these changed models files.

This module does two things:
1. It updates older saved models during loading.

2. It warns if a model was created with a newer PCNtoolkit version than
the one currently installed by the user.

In simple:
version_model < version_pcntoolkit  →  APPLY MIGRATIONS
version_model = version_pcntoolkit  →  DO NOTHING
version_model > version_pcntoolkit  →  WARN USER
"""

# ---------------------------------------------------------------------------
# MigrationRegistry class to apply the migration functions defined above.
# ---------------------------------------------------------------------------

from __future__ import annotations

import importlib.metadata
from typing import Callable, Literal

# Use the "packaging" module to read also "post1" versions
from packaging.version import Version

from pcntoolkit.util.output import Output, Warnings

# All components that are saved in the model file and may require migrations.
ComponentName = Literal[
    "BLR",
    "HBR",
    "BasisFunction",
    "Scaler",
    "Likelihood",
    "Prior",
]


class MigrationRegistry:
    """
    Registers and applies model migration functions.

    Migration functions update older saved model dictionaries to
    the format expected by the current PCNtoolkit version.

    Migrations are applied automatically in version order when a
    model is loaded.

    Attributes
    ----------
    _migrations : dict[ComponentName, list[tuple[Version, Callable]]]
        Maps component name -> sorted list of
        (introduced_in_version, migration_fn) tuples.
    """

    def __init__(self) -> None:
        # Maps component name -> sorted list of
        # (introduced_in_version, migration_fn) tuples.
        self._migrations: dict[
            ComponentName, list[tuple[Version, Callable[[dict], dict]]]
        ] = {}

    def register(
        self,
        component: ComponentName,
        introduced_in: str,
    ) -> Callable[[Callable[[dict], dict]], Callable[[dict], dict]]:
        """
        Decorator used to register a migration function for a component.

        When a function is decorated with @registry.register(...), it is
        automatically added to self._migrations.

        Parameters
        ----------
        component : ComponentName
            Name of the component being migrated
            (e.g. "BLR", "BasisFunction", "Scaler").
        introduced_in : str
            The PCNtoolkit version in which the new format was
            introduced.

        Returns
        -------
        Callable
            The migration function.
        """
        # Convert the version string to a comparable Version object.
        target_version: Version = Version(introduced_in)

        def decorator(
            fn: Callable[[dict], dict],
        ) -> Callable[[dict], dict]:
            # Add to the registry list for this component.
            if component not in self._migrations:
                self._migrations[component] = []
            self._migrations[component].append((target_version, fn))
            # Keep migrations sorted by version so they run in order.
            self._migrations[component].sort(key=lambda t: t[0])
            return fn

        return decorator

    def migrate(
        self,
        component: ComponentName,
        d: dict,
        version: str | None = None,
    ) -> dict:
        """
        Apply all migrations specified in self._migrations when loading models
        saved with previous PCNtoolkit versions.

        Called by from_dict() methods that exist in the components being
        migrated (e.g. BasisFunction.from_dict()).

        Parameters
        ----------
        component : ComponentName
            Name of the component (must match what was used in
            register()).
        d : dict
            The raw dict read from a saved JSON file.
        version : str | None, optional
            Explicit version override. If no version is exists in the JSON
            file, it defaults to 0.0.0

        Returns
        -------
        dict
            The dict, updated to the format expected by the current
            PCNtoolkit version.
        """
        # Determine the version of the saved dict.
        raw_version: str = (
            version if version is not None else d.get("ptk_version", "0.0.0")
        )
        # Use "0.0.0" as fallback for models saved before versioning.
        saved_version: Version = Version(raw_version or "0.0.0")

        # If no migrations are registered for this component, do nothing.
        if component not in self._migrations:
            return d

        # The version the user is actually running, not the version a
        # migration was introduced in.
        current_version: str = importlib.metadata.version("pcntoolkit")

        # Apply migrations in ascending version order.
        for introduced_in, fn in self._migrations[component]:
            if saved_version < introduced_in:
                # Emit an informational message that migration is running.
                Output.warning(
                    Warnings.MODEL_MIGRATION_APPLIED,
                    saved_version=str(saved_version),
                    current_version=current_version,
                )
                d = fn(d)

        return d


def check_forward_compatibility(
    saved_version: str,
    current_version: str,
) -> None:
    """
    Warn if a model was created with a newer PCNtoolkit version.

    Newer model files may contain features or formats that are not
    supported by the older installed version.

    Parameters
    ----------
    saved_version : str
        The ptk_version string stored in the model file.
    current_version : str
        The version of the currently installed pcntoolkit package.

    Returns
    -------
    None
    """
    # Default to "0.0.0" when version is missing (old models).
    parsed_saved: Version = Version(saved_version or "0.0.0")
    parsed_current: Version = Version(current_version or "0.0.0")

    if parsed_saved > parsed_current:
        # Emit a warning so the user knows to update.
        Output.warning(
            Warnings.MODEL_SAVED_WITH_NEWER_VERSION,
            saved_version=str(parsed_saved),
            current_version=str(parsed_current),
        )


# MigrationRegistry is a singleton: All components import this same instance
# of registry which holds all registered migration functions.
registry: MigrationRegistry = MigrationRegistry()

# ---------------------------------------------------------------------------
# Add migration functions below
# ---------------------------------------------------------------------------


@registry.register("BasisFunction", introduced_in="1.2.0post1")
def _migrate_basis_function_1_2_0post1(d: dict) -> dict:
    """Migrate a BasisFunction dict from v1.1.2 to v1.2.0post1.

    Three parameters changed representation. See issue #435.

    | Parameter    | v1.1.2               | v1.2.0post1                   |
    |--------------|----------------------|-------------------------------|
    | basis_column | `[0]` (list)         | `0` (int)                     |
    | min / max    | `{"0": -2.74}` (dict)| `-2.74` (float)               |
    | knots        | `{"0": [...]}` (dict)| `[...]` (list)                |

    Not implemented
    -------------
    Basis functions with multiple basis columns saved before v1.2.0post1
    could store knots, min and max as multi-key dicts
    (e.g. ``{"0": [...], "1": [...]}``).
    Migrating these to version higher than v1.2.0post1 is not yet supported.
    A ``NotImplementedError`` is raised when such a model is loaded.

    Parameters
    ----------
    d : dict
        Raw dict read from the saved JSON file.

    Returns
    -------
    dict
        Dict with all three parameters changed to the v1.2.0post1
        format.
    """
    # Raise an error if this is a basis function with multiple basis columns
    for key in ("min", "max", "knots"):
        if isinstance(d.get(key), dict) and len(d[key]) > 1:
            raise NotImplementedError(
                "Loading a basis function with multiple basis columns saved "
                "with PCNtoolkit v1.1.2 or earlier is not supported."
            )

    # Fix 1: basis_column: list -> int
    if isinstance(d.get("basis_column"), list):
        d["basis_column"] = d["basis_column"][0]

    # Fix 2: min / max: {"0": value} dicts -> floats.
    if isinstance(d.get("min"), dict):
        if d["min"]:
            d["min"] = next(iter(d["min"].values()))
        else:
            # default to None if the dict is empty (e.g. {} in JSON).
            d["min"] = None
    if isinstance(d.get("max"), dict):
        if d["max"]:
            d["max"] = next(iter(d["max"].values()))
        else:
            # default to None if the dict is empty (e.g. {} in JSON).
            d["max"] = None

    # Fix 3: knots: {"0": [...]} dict -> list.
    if isinstance(d.get("knots"), dict):
        if d["knots"]:
            d["knots"] = next(iter(d["knots"].values()))
        else:
            # default to None if the dict is empty (e.g. {} in JSON).
            d["knots"] = None

    return d


@registry.register("HBR", introduced_in="1.4.0")
def _migrate_variational_inference_1_4_0(d: dict) -> dict:
    """Add the variational inference fields to an HBR dict saved before VI.
    These fields specify what methods is used for fitting
    (e.g., "mcmc", "pathfinder" etc).

    Parameters
    ----------
    d : dict
        Raw dict read from the saved JSON file.

    Returns
    -------
    dict
        Dict with the variational inference fields populated.
    """
    d.setdefault("inference_method", "mcmc")
    d.setdefault("vi_iterations", 30000)
    d.setdefault("vi_draws", 1000)
    d.setdefault("vi_kwargs", {})

    return d


@registry.register("BasisFunction", introduced_in="1.4.0")
def _migrate_bspline_basis_function_1_4_0(d: dict) -> dict:
    """Preserve legacy B-spline behavior for models saved before v1.4.0.

    Before v1.4.0, BsplineBasisFunction implicitly included the original
    linear feature alongside the B-spline basis. New B-spline basis
    functions no longer include this redundant linear term.

    When loading an older saved B-spline model, retain the old behavior
    so that its design matrix remains compatible with the fitted model
    parameters.
    """
    if d.get("basis_function") == "BsplineBasisFunction":
        d.setdefault("include_linear", True)

    return d


@registry.register("BLR", introduced_in="1.4.0")
def _migrate_blr_slope_indices_1_4_0(d: dict) -> dict:
    """Keep the batch effect slopes of BLR models saved before v1.4.0 on covariate 0.

    Before v1.4.0, `fixed_effect_slope_indices=None` (and the variance
    counterpart) always meant `[0]`. From v1.4.0, None means the covariate
    that the basis function expands (its `basis_column`). Setting `[0]`
    explicitly keeps old models identical when that column is not 0.

    `fixed_effect_slope_indices` > 0 pointed at basis function columns before v1.4.0
    and at covariates from v1.4.0 on. So indices other than None
    or [0] are rejected from 1.4.0 onwards as they have a different meaning than before.

    Parameters
    ----------
    d : dict
        Raw dict read from the saved JSON file.

    Returns
    -------
    dict
        Dict with explicit slope indices.

    Raises
    ------
    NotImplementedError
        If a non-linear basis has slope indices other than None or [0].
    """
    for slope, key, basis in (
        ("fixed_effect_slope", "fixed_effect_slope_indices", "basis_function_mean"),
        (
            "fixed_effect_var_slope",
            "fixed_effect_var_slope_indices",
            "basis_function_var",
        ),
    ):
        basis_name = (d.get(basis) or {}).get("basis_function")
        indices = d.get(key)
        # non linear are the Polynomial or Bspline basis functions
        nonlinear = basis_name != "LinearBasisFunction"
        if d.get(slope) and nonlinear and indices not in (None, [0]):
            raise NotImplementedError(
                f"Cannot load this model: it uses {key}={indices} with a "
                f"{basis_name}. Before PCNtoolkit v1.4.0, these indices counted "
                "the columns made by the basis function (e.g. 1 = the first "
                "B-spline column). From v1.4.0 they count the covariates "
                "(e.g. 1 = the second covariate). Refit the model with "
                "PCNtoolkit v1.4.0 or later, or load it with pcntoolkit<1.4.0."
            )

    for key in ("fixed_effect_slope_indices", "fixed_effect_var_slope_indices"):
        if d.get(key) is None:
            d[key] = [0]

    return d
