# MIGRATION GUIDE

This guide explains how PCNtoolkit handles models that were
saved with an older or newer version of the library. You do
not need to be a software engineer to understand it.

---

## Why is this needed?

As PCNtoolkit is developed, the way models store their
settings in JSON files sometimes has to change. For example,
a parameter might be renamed or a new field might be added.

Without a migration system, a model saved six months ago
might fail to load today because the code expects a field
that did not exist back then.

The migration system solves this automatically: when you
load an old model, the library detects the mismatch and
silently upgrades the saved data to the current format
before loading it.

---

## What is `ptk_version`?

Every model file saved by PCNtoolkit contains a `ptk_version`
field — for example:

```json
{
  "ptk_version": "1.2.0",
  "name": "my_model",
  ...
}
```

This records which version of the library created the file.
When you load a model, the library compares this version
against the version you have installed.

- **Older saved version** → any necessary migrations are
  applied automatically, and you will see an informational
  message for each one.
- **Newer saved version** → you will see a warning telling
  you to upgrade PCNtoolkit. The model may still load, but
  some features might not work correctly.

---

## What happens automatically when I load a model?

Nothing special is required from you. Simply call:

```python
model = NormativeModel.load("path/to/saved_model")
```

The library will:

1. Read `ptk_version` from the JSON file.
2. Compare it to the currently installed version.
3. Warn you if the file was saved with a **newer** version.
4. Apply any registered migrations if the file was saved
   with an **older** version.

---

## I am a developer — how do I register a migration?

Whenever you make a change to a JSON-serialised dict that
breaks backward compatibility (renaming a key, removing a
field, changing its type), you **must** register a migration.

### Step 1: Identify the component name

Each serialisable class has a component name used in the
registry. Current names are:

| Class            | Component name    |
|------------------|-------------------|
| `BLR`            | `"BLR"`           |
| `HBR`            | `"HBR"`           |
| `BasisFunction`  | `"BasisFunction"` |
| `Scaler`         | `"Scaler"`        |
| `Likelihood`     | `"Likelihood"`    |
| `BasePrior` etc. | `"Prior"`         |

### Step 2: Write and register the migration

Open `pcntoolkit/util/migration.py` and add your function
at the bottom of the file, under the
`# Registered migrations` comment block:

```python
@registry.register("BasisFunction", introduced_in="1.3.0")
def _migrate_basis_function_1_3_0(d: dict) -> dict:
    """Rename 'knot_positions' to 'knots' (changed in 1.3.0)."""
    # Only rename if the old key is present (safe to call on
    # already-migrated dicts).
    if "knot_positions" in d:
        d["knots"] = d.pop("knot_positions")
    return d
```

**Rules:**
- `introduced_in` must be the version in which the **new**
  format first appears (i.e. the version you are about to
  release, not the old one).
- The migration function receives the raw `dict` and must
  return a (modified) `dict`. Always make changes
  conditional (`if "old_key" in d`) so the function is safe
  to call on dicts that were already migrated.
- Choose a descriptive function name:
  `_migrate_<component>_<version_with_underscores>`.
- Add a one-line docstring explaining what changed.

### Step 3: Bump the version

Increment `version` in `pyproject.toml` to match the
`introduced_in` you used.

### Step 4: Add a test

Add a test in `test/test_util/test_migration.py` to verify
that the migration transforms the old dict correctly. Use a
local `MigrationRegistry()` instance (not the module
singleton) so the test is isolated.

---

## Version comparison rules

PCNtoolkit uses [PEP 440][pep440] version strings and the
`packaging` library to compare versions. This means
post-releases (`1.2.0.post1`), release candidates (`1.3.0rc1`),
and development versions (`1.3.0.dev0`) are all handled
correctly:

```
1.2.0.post1 > 1.2.0
1.3.0rc1 < 1.3.0
1.3.0.dev0 < 1.3.0
```

Models saved without a `ptk_version` field (very old models)
are treated as version `0.0.0`, so all registered migrations
will be applied to them.

[pep440]: https://peps.python.org/pep-0440/
