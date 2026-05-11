"""
Model migration registry for PCNtoolkit.

When a saved model's serialized dict format changes between
versions, a migration function can be registered here to
automatically upgrade old dicts to the current format when
loading.

Usage for developers
--------------------
To add a migration for a breaking change introduced in version
X.Y.Z, decorate a function with:

    @registry.register("ComponentName", introduced_in="X.Y.Z")
    def migrate_componentname_x_y_z(d: dict) -> dict:
        # Transform d from the old format to the new format.
        # Example: rename a key
        if "old_key" in d:
            d["new_key"] = d.pop("old_key")
        return d

"ComponentName" must match the string passed to
registry.migrate(...) in the corresponding from_dict() method.
"""
from __future__ import annotations

import importlib.metadata
from typing import Callable

from packaging.version import Version

from pcntoolkit.util.output import Output, Warnings


class MigrationRegistry:
    """
    Central registry of dict-migration functions.

    Migration functions are keyed by (component, introduced_in)
    and are applied in version order when loading a saved dict
    whose ptk_version is older than the current code.

    Attributes
    ----------
    _migrations : dict[str, list[tuple[Version, Callable]]]
        Maps component name -> sorted list of
        (introduced_in_version, migration_fn) tuples.
    """

    def __init__(self) -> None:
        # Maps component name to list of (version, fn) tuples.
        self._migrations: dict[
            str, list[tuple[Version, Callable[[dict], dict]]]
        ] = {}

    def register(
        self,
        component: str,
        introduced_in: str,
    ) -> Callable[[Callable[[dict], dict]], Callable[[dict], dict]]:
        """
        Decorator that registers a migration for a component.

        Parameters
        ----------
        component : str
            Name of the component being migrated (e.g. "BLR",
            "BasisFunction", "Scaler").
        introduced_in : str
            The PCNtoolkit version in which the new format was
            introduced. Models saved with an older version than
            this will have the migration applied.

        Returns
        -------
        Callable
            The original function, unchanged.

        Examples
        --------
        @registry.register("BLR", introduced_in="1.3.0")
        def migrate_blr_1_3_0(d: dict) -> dict:
            if "old_key" in d:
                d["new_key"] = d.pop("old_key")
            return d
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
        component: str,
        d: dict,
        version: str | None = None,
    ) -> dict:
        """
        Apply all pending migrations to a dict.

        Reads the saved version from d["ptk_version"] (or uses
        version if provided, defaulting to "0.0.0" if absent).
        Applies every registered migration for this component
        where saved_version < introduced_in, in ascending order.

        This method is always safe to call: it is a no-op when no
        migrations are registered or the dict is already current.

        Parameters
        ----------
        component : str
            Name of the component (must match what was used in
            register()).
        d : dict
            The raw dict read from a saved JSON file.
        version : str | None, optional
            Explicit version override. If None, reads
            d.get("ptk_version", "0.0.0").

        Returns
        -------
        dict
            The (potentially updated) dict.
        """
        # Determine the version of the saved dict.
        raw_version: str = (
            version
            if version is not None
            else d.get("ptk_version", "0.0.0")
        )
        # Use "0.0.0" as fallback for models saved before versioning.
        saved_version: Version = Version(raw_version or "0.0.0")

        # If no migrations are registered for this component, do nothing.
        if component not in self._migrations:
            return d

        # Apply migrations in ascending version order.
        for introduced_in, fn in self._migrations[component]:
            if saved_version < introduced_in:
                # Emit an informational message that migration is running.
                Output.warning(
                    Warnings.MODEL_MIGRATION_APPLIED,
                    component=component,
                    from_version=str(saved_version),
                    to_version=str(introduced_in),
                )
                d = fn(d)

        return d


def check_forward_compatibility(
    saved_version: str,
    current_version: str,
) -> None:
    """
    Warn the user if a saved model was created with a newer
    version of PCNtoolkit than the currently installed one.

    Models created with a newer version may use features that
    are not present in the running code. The user should update
    PCNtoolkit to avoid potential errors.

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


# ---------------------------------------------------------------------------
# Module-level singleton registry — import this in from_dict() methods.
# ---------------------------------------------------------------------------
registry: MigrationRegistry = MigrationRegistry()


# ---------------------------------------------------------------------------
# Registered migrations
# ---------------------------------------------------------------------------
# Add migration functions below using the @registry.register decorator.
# Each function must accept a dict and return a (modified) dict.
# Keep one function per breaking change; name it clearly.
#
# Example:
#
#   @registry.register("BasisFunction", introduced_in="1.3.0")
#   def _migrate_basis_function_1_3_0(d: dict) -> dict:
#       """Rename 'knot_positions' to 'knots' (changed in 1.3.0)."""
#       if "knot_positions" in d:
#           d["knots"] = d.pop("knot_positions")
#       return d
#
# ---------------------------------------------------------------------------
