"""GitHub source link resolver for sphinx.ext.linkcode.

Imported by conf.py so that ``linkcode_resolve`` is available in the
Sphinx build without cluttering the main configuration file.
"""

import importlib  # resolve dotted module paths
import inspect    # extract source file and line numbers
import os


def linkcode_resolve(
    domain: str, info: dict[str, str]
) -> str | None:
    """Map a documented Python object to its GitHub source URL.

    Called by sphinx.ext.linkcode for every documented object.
    Returns a permalink to the exact line on GitHub, or None when
    the source location cannot be determined so Sphinx skips the
    link gracefully.

    Parameters
    ----------
    domain : str
        Sphinx domain being processed; only 'py' is handled.
    info : dict[str, str]
        Must contain 'module' (dotted import path) and 'fullname'
        (object name, possibly dotted for nested members).

    Returns
    -------
    str | None
        GitHub blob URL ending in ``#L<lineno>``, or None.
    """
    # Only Python objects have source links worth generating
    if domain != "py":
        return None
    # Both keys must be non-empty strings to locate the object
    if not info.get("module") or not info.get("fullname"):
        return None

    # Determine the branch/tag to link to in the GitHub URL.
    # ReadTheDocs sets READTHEDOCS_VERSION; map symbolic names
    # ('latest', 'stable') and local builds to the master branch.
    rtd_version = os.environ.get("READTHEDOCS_VERSION", "")
    if rtd_version in ("latest", "stable", ""):
        # Local builds and symbolic RTD versions → master
        branch = "master"
    else:
        # Specific release tag (e.g. "1.0.0") → link to that tag
        branch = rtd_version

    # Repo root is one directory above doc/ (where conf.py lives)
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    # Import the module that declares the object
    try:
        mod = importlib.import_module(info["module"])
    except ImportError:
        return None

    # Navigate the dotted fullname to reach the innermost object
    obj: object = mod
    for part in info["fullname"].split("."):
        try:
            obj = getattr(obj, part)  # step into each attribute
        except AttributeError:
            return None

    # Unwrap decorators so inspect sees the original function;
    # guard against pathological __wrapped__ cycles
    try:
        obj = inspect.unwrap(obj)
    except StopIteration:
        pass

    # Resolve the source file and the first line of the object
    try:
        src_file = inspect.getsourcefile(obj)
        if src_file is None:
            # Built-in or C-extension: no Python source available
            return None
        _, start_line = inspect.getsourcelines(obj)
    except (TypeError, OSError):
        # inspect cannot determine source for some objects
        return None

    # Build a path relative to the repo root so the URL is stable
    try:
        rel_path = os.path.relpath(src_file, repo_root)
    except ValueError:
        # On Windows, relpath fails when paths span different drives
        return None

    # Normalise Windows path separators to URL forward slashes
    rel_path = rel_path.replace("\\", "/")

    # Assemble the full GitHub permalink with line-number anchor
    base = (
        "https://github.com/predictive-clinical-neuroscience"
        "/PCNtoolkit"
    )
    return f"{base}/blob/{branch}/{rel_path}#L{start_line}"
