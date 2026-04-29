"""GitHub source link resolver for sphinx.ext.linkcode.

Imported by conf.py so that ``linkcode_resolve`` is available in the
Sphinx build without cluttering the main configuration file.

Uses AST-based source scanning so the package does not need to be
pip-installed during the build (required for ReadTheDocs builds that
scan sources without installing the library).
"""

import ast      # parse Python source files without importing them
import os


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _module_to_path(module: str, repo_root: str) -> str | None:
    """Convert a dotted module name to an absolute file path.

    Performs a pure string conversion: dots become path separators and
    ``.py`` is appended.  The file must exist under *repo_root*.

    Parameters
    ----------
    module : str
        Dotted module name, e.g. ``pcntoolkit.normative_model``.
    repo_root : str
        Absolute path to the repository root.

    Returns
    -------
    str | None
        Absolute path to the ``.py`` file, or ``None`` if not found.
    """
    # Replace dots with OS separators to get a relative path fragment
    rel = os.path.join(*module.split(".")) + ".py"
    # Candidate 1: module maps to a plain .py file
    candidate = os.path.join(repo_root, rel)
    if os.path.isfile(candidate):
        return candidate
    # Candidate 2: module is a package (directory with __init__.py)
    pkg = os.path.join(
        repo_root,
        os.path.join(*module.split(".")),
        "__init__.py",
    )
    if os.path.isfile(pkg):
        return pkg
    return None  # file not found in the repo


def _find_lineno(src_path: str, fullname: str) -> int | None:
    """Find the first line of a named object in a Python source file.

    Parses *src_path* with the ``ast`` module (no import needed) and
    searches for a class or function definition matching the first
    component of *fullname*.  Nested members (e.g. ``MyClass.method``)
    are not resolved to an inner line; the class line is returned
    instead so the link still lands close to the right place.

    Parameters
    ----------
    src_path : str
        Absolute path to the ``.py`` source file.
    fullname : str
        Object name as provided by sphinx.ext.linkcode, e.g.
        ``NormativeModel`` or ``NormativeModel.fit``.

    Returns
    -------
    int | None
        1-based line number of the definition, or ``None`` on failure.
    """
    # Only look at the top-level name (first segment before any dot)
    top_name = fullname.split(".")[0]
    try:
        # Read and parse the source file into an AST
        with open(src_path, encoding="utf-8") as fh:
            source = fh.read()
        tree = ast.parse(source, filename=src_path)
    except (OSError, SyntaxError):
        return None  # unreadable or invalid Python

    # Walk top-level nodes only; no need for a deep traversal
    for node in ast.iter_child_nodes(tree):
        if isinstance(
            node,
            (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            # Compare the node name to the target top-level name
            if node.name == top_name:
                return node.lineno  # 1-based line number from AST
    return None  # name not found at module top level


# ---------------------------------------------------------------------------
# Public API required by sphinx.ext.linkcode
# ---------------------------------------------------------------------------

def linkcode_resolve(
    domain: str, info: dict[str, str]
) -> str | None:
    """Map a documented Python object to its GitHub source URL.

    Called by sphinx.ext.linkcode for every documented object.
    Returns a permalink to the exact line on GitHub, or ``None`` when
    the source location cannot be determined so Sphinx skips the
    link gracefully.

    Works without importing the package: the source file is located by
    converting the dotted module name to a file path, and the line
    number is found by parsing the file with ``ast``.  This makes it
    safe to use on ReadTheDocs builds that do not install pcntoolkit.

    Parameters
    ----------
    domain : str
        Sphinx domain being processed; only ``'py'`` is handled.
    info : dict[str, str]
        Must contain ``'module'`` (dotted import path) and
        ``'fullname'`` (object name, possibly dotted for nested
        members).

    Returns
    -------
    str | None
        GitHub blob URL ending in ``#L<lineno>``, or ``None``.
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

    # Resolve the dotted module name to an absolute file path
    src_path = _module_to_path(info["module"], repo_root)
    if src_path is None:
        return None  # source file not found in repo

    # Find the line number using AST (no import required)
    lineno = _find_lineno(src_path, info["fullname"])
    if lineno is None:
        # Fall back to line 1 so the link still points at the file
        lineno = 1

    # Build a path relative to the repo root so the URL is stable
    try:
        rel_path = os.path.relpath(src_path, repo_root)
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
    return f"{base}/blob/{branch}/{rel_path}#L{lineno}"

