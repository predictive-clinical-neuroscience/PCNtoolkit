"""
Unit tests for the MigrationRegistry and
check_forward_compatibility function in util/migration.py.

Tests use local registry instances so they are fully isolated
from the module-level singleton and from each other.
"""

from unittest.mock import patch

from pcntoolkit.util.migration import (
    MigrationRegistry,
    check_forward_compatibility,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry_with_two_migrations() -> MigrationRegistry:
    """Return a fresh registry with two sequential migrations."""
    # Create an isolated registry for the test.
    reg = MigrationRegistry()

    # Migration introduced in 1.1.0. It renames "old_key" -> "new_key".
    @reg.register("TestComp", introduced_in="1.1.0")
    def _migrate_1_1_0(d: dict) -> dict:
        if "old_key" in d:
            d["new_key"] = d.pop("old_key")
        return d

    # Migration introduced in 1.2.0. It adds a default field.
    @reg.register("TestComp", introduced_in="1.2.0")
    def _migrate_1_2_0(d: dict) -> dict:
        d.setdefault("new_key", 42)
        return d

    return reg


# ---------------------------------------------------------------------------
# MigrationRegistry.migrate tests
# ---------------------------------------------------------------------------

def test_001_migrate_should_applyMigration_when_savedVersionIsOlder():
    """
    Arrange: dict saved with version 1.0.0 (before 1.1.0 migration).
    Act: call registry.migrate with that version.
    Assert: "old_key" is renamed to "new_key".
    """
    # Arrange — dict from a model saved with version 1.0.0.
    reg = _make_registry_with_two_migrations()
    d = {"old_key": "value"}

    # Act — migrate from version 1.0.0.
    result = reg.migrate("TestComp", d, version="1.0.0")

    # Assert — key has been renamed.
    assert "new_key" in result


def test_002_migrate_should_notApplyMigration_when_savedVersionIsCurrent():
    """
    Arrange: dict saved with version 1.2.0 (same as last migration).
    Act: call registry.migrate with that version.
    Assert: dict is unchanged (no migration applied).
    """
    # Arrange — dict from a model saved at current version 1.2.0.
    reg = _make_registry_with_two_migrations()
    d = {"some_key": "value"}

    # Act — migrate from version 1.2.0.
    result = reg.migrate("TestComp", d, version="1.2.0")

    # Assert — no added_key injected (migration not applied).
    assert "added_key" not in result


def test_003_migrate_should_applyAllMigrationsInOrder_when_multipleVersionsBehind():
    """
    Arrange: dict saved with version 1.0.0 (two migrations behind).
    Act: call registry.migrate with that version.
    Assert: both changes are applied.
    """
    # Arrange — old dict missing both changes.
    reg = _make_registry_with_two_migrations()
    d = {"old_key": "value"}

    # Act — migrate from version 1.0.0.
    result = reg.migrate("TestComp", d, version="1.0.0")

    # Assert — both migrations have run in order.
    assert "new_key" in result
    assert result.get("new_key") == 42


def test_004_migrate_should_defaultToZeroVersion_when_ptk_versionMissing():
    """
    Arrange: dict with no ptk_version key (very old model).
    Act: call registry.migrate without providing version.
    Assert: all migrations are applied (treated as 0.0.0).
    """
    # Arrange — dict with no ptk_version key.
    reg = _make_registry_with_two_migrations()
    d = {"old_key": "value"}

    # Act — no explicit version, should default to "0.0.0".
    result = reg.migrate("TestComp", d)

    # Assert — both migrations ran because saved < 1.1.0 and < 1.2.0.
    assert "new_key" in result
    assert result.get("new_key") == 42


def test_005_migrate_should_notApplyMigration_when_UnknownComponent():
    """
    Arrange: a registry with no migrations for "UnknownComp".
    Act: call migrate for "UnknownComp".
    Assert: the dict is returned unchanged.
    """
    # Arrange — empty registry, unknown component.
    reg = MigrationRegistry()
    d = {"key": "value"}

    # Act — migrate an unregistered component.
    result = reg.migrate("UnknownComp", d, version="1.0.0")

    # Assert — dict is unchanged.
    assert result == {"key": "value"}


# ---------------------------------------------------------------------------
# check_forward_compatibility tests
# ---------------------------------------------------------------------------

def test_006_checkForwardCompat_should_emitWarning_when_savedVersionIsNewer():
    """
    Arrange: saved version 2.0.0, current version 1.0.0.
    Act: call check_forward_compatibility.
    Assert: Output.warning is called once.
    """
    # Arrange — saved version is ahead of current version.
    saved_model = "2.0.0"
    current_pcntoolkit = "1.0.0"

    # Act & Assert — warning must be emitted exactly once.
    with patch(
        "pcntoolkit.util.migration.Output.warning"
    ) as mock_warning:
        check_forward_compatibility(saved_model, current_pcntoolkit)
        assert mock_warning.call_count == 1


def test_007_checkForwardCompat_should_notEmitWarning_when_versionsSame():
    """
    Arrange: saved version equals current version.
    Act: call check_forward_compatibility.
    Assert: Output.warning is not called.
    """
    # Arrange — both versions are identical.
    saved_model = "1.2.0"
    current_pcntoolkit = "1.2.0"

    # Act & Assert — no warning should be emitted.
    with patch(
        "pcntoolkit.util.migration.Output.warning"
    ) as mock_warning:
        check_forward_compatibility(saved_model, current_pcntoolkit)
        assert mock_warning.call_count == 0


def test_008_checkForwardCompat_should_notEmitWarning_when_savedVersionIsOlder():
    """
    Arrange: saved version 1.0.0, current version 1.2.0.
    Act: call check_forward_compatibility.
    Assert: Output.warning is not called.
    """
    # Arrange — saved version is behind current version.
    saved_model = "1.0.0"
    current_pcntoolkit = "1.2.0"

    # Act & Assert — no warning should be emitted.
    with patch(
        "pcntoolkit.util.migration.Output.warning"
    ) as mock_warning:
        check_forward_compatibility(saved_model, current_pcntoolkit)
        assert mock_warning.call_count == 0
