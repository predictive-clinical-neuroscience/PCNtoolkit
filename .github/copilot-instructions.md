# 1. Purpose of this File

This file gives AI coding agents and contributors concise,
repo-specific guidance to work effectively: how to set up the
environment, information about the software architecture, follow
house rules (style, commits), and avoid common pitfalls. When you create new code or change old code you MUST explain everything you do clearly and simply for the developer to understand, and you MUST run tests that are related to this change.

# 2. High-level Repository Summary

- Purpose: PCNtoolkit (Predictive Clinical Neuroscience Toolkit)
  — a Python library for normative modelling of neuroimaging
  data, supporting Bayesian Linear Regression (BLR) and
  Hierarchical Bayesian Regression (HBR) with multiple
  likelihood functions.
- Target users: Neuroscientists, clinicians, and researchers
  performing normative modelling; developers extending
  regression models, likelihoods, or data pipelines.
- Tech stack:
  - Python >=3.11, <3.13
  - PyMC / nutpie for Bayesian inference
  - NumPy, SciPy, scikit-learn, pandas, xarray for data
  - matplotlib, seaborn for plotting
  - nibabel for neuroimaging I/O
  - joblib, dill, cloudpickle for parallelism/serialization
  - dask for lazy computation
  - Sphinx + AutoAPI for documentation
  - pytest for testing
- Repo type: Library (pip-installable package) with Jupyter
  notebook examples and Sphinx documentation.
- Design: Scientific computing library organised around a
  `NormativeModel` that wraps a `RegressionModel` (BLR or HBR),
  with composable basis functions, scalers, likelihoods, and
  priors. A `Runner` class handles cluster/parallel job
  submission.

# 3. Key File Locations and What They Contain

- Configuration
  - `pyproject.toml` — project metadata, dependencies, build
    config, ruff settings, and entry points
  - `pytest.ini` — pytest configuration
  - `.gitignore` — ignores build artifacts, test resources,
    large files
- Entry points
  - `pcntoolkit/normative.py` — CLI entry point (`normative`
    command); functions `fit`, `predict`, `fit_predict`, `main`,
    `entrypoint`
  - `pcntoolkit/__init__.py` — public API re-exports
- Core library (`pcntoolkit/`)
  - `normative_model.py` — `NormativeModel` class: top-level
    model orchestrator (fit, predict, transfer, evaluate)
  - `regression_model/` — regression backends:
    - `blr.py` — `BLR` (Bayesian Linear Regression)
    - `hbr.py` — `HBR` (Hierarchical Bayesian Regression)
    - `regression_model.py` — `RegressionModel` base class
    - `factory.py` — factory to create regression models from
      config
    - `test_model.py` — `TestModel` for unit testing
  - `math_functions/` — mathematical building blocks:
    - `basis_function.py` — `BasisFunction` ABC and concrete
      implementations (`Linear`, `Polynomial`, `Bspline`,
      `Composite`)
    - `likelihood.py` — likelihood classes (`Normal`, `Beta`,
      `SHASHb`)
    - `prior.py` — `Prior`, `BasePrior`, `make_prior`,
      prior-transfer logic
    - `scaler.py` — `Scaler` classes (`StandardScaler`, etc.)
    - `warp.py` — warping functions for BLR
    - `factorize.py` — factorization helpers for prior transfer
    - `shash.py` — SHASH distribution (PyMC custom dist)
  - `dataio/` — data I/O:
    - `norm_data.py` — `NormData` class (xarray-based data
      container)
    - `fileio.py` — file reading/writing (CSV, NIFTI, CIFTI)
    - `data_factory.py` — `load_fcon1000` and dataset helpers
  - `util/` — utilities:
    - `runner.py` — `Runner` class for cluster job management
      (SLURM/Torque)
    - `evaluator.py` — `Evaluator` for model evaluation metrics
    - `output.py` — `Output`, `Messages`, `Warnings`, `Errors`
      classes for structured logging/messaging
    - `paths.py` — default directory resolution helpers
    - `plotter.py` — plotting functions (`plot_centiles`,
      `plot_qq`, `plot_ridge`, `plot_centiles_advanced`)
    - `job_observer.py` — `JobObserver` for monitoring cluster
      jobs
    - `model_comparison.py` — model comparison utilities
    - `autoscale_plot.py` — auto-scaling plot helper
- Tests (`test/`)
  - `conftest.py` — shared pytest configuration
  - `fixtures/` — reusable pytest fixtures (data, models, paths)
  - `test_cli/` — CLI integration tests
  - `test_core/` — core functionality tests
  - `test_dataio/` — NormData tests
  - `test_math/` — basis function, factorize, warp tests
  - `test_norm/` — normative model tests
  - `test_regression_models/` — BLR/HBR tests
  - `test_util/` — runner, MSLL, utility tests
- Examples (`examples/`)
  - Jupyter notebooks `00_getting_started.ipynb` through
    `10_merge.ipynb` demonstrating usage
  - `big_cluster_jobs.py` — example cluster submission script
- Documentation (`doc/`)
  - Sphinx source: `conf.py`, `index.rst`, `api.rst`
  - Tutorial RST files in `doc/pages/tutorials/`
  - `convert_notebooks.py` — converts notebooks to RST
- Other
  - `render_citation.py` — generates `CITATION.cff` from
    template
  - `Makefile` — dev-setup target (conda env + pip install)
  - `CONTRIBUTING.md` — contribution guidelines
  - `CHANGES` — changelog

# 4. Dependencies and Environment

- System requirements
  - Python >=3.11, <3.13; Git
  - On Windows: conda packages `m2w64-toolchain` and
    `libpython` are needed for C extensions
- Package manager
  - pip; conda recommended for environment management
- Core dependencies (from `pyproject.toml`)
  - nibabel, pymc, scikit-learn, scipy, matplotlib, seaborn,
    numba, nutpie, joblib, dill, ipywidgets, ipykernel, dask,
    filelock, six, cloudpickle (via pymc)
- Dev dependencies
  - toml, sphinx-tabs, pytest, black, sphinx-rtd-theme, nbsphinx,
    nbconvert, pandoc, jinja2, autoapi, ruff, pytest-cov
- Environment variables
  - `PCN_HOME_DIR` — override default home directory
    (`~/.pcntoolkit`)
  - `PCN_SAVE_DIR` — override default save directory
  - `PCN_TEMP_DIR` — override default temp directory
  - `PCN_LOG_DIR` — override default log directory
- Platform notes
  - Cross-platform (Linux, macOS, Windows)
  - Cluster support for SLURM and Torque via `Runner`

# 5. Development Workflow

- Initial setup
  ```bash
  conda create -n ptk-dev python=3.12 -y
  conda activate ptk-dev
  # Windows only:
  conda install -c conda-forge m2w64-toolchain libpython
  pip install -e ".[dev]"
  ```
  Or use the Makefile:
  ```bash
  make dev-setup
  ```
- Run tests
  ```bash
  pytest
  ```
  Or with coverage:
  ```bash
  pytest --cov=pcntoolkit
  ```
- Build documentation
  ```bash
  cd doc
  make html
  ```
- Lint/format
  Use autopep8 or ruff:

  ```bash
  ruff check pcntoolkit/
  ruff format pcntoolkit/
  ```

- Validate a change
  - Run `pytest` and ensure all tests pass
  - Run `ruff check` and fix any issues
  - Test affected notebooks if data pipeline changed

# 6. Code Style and Standards

- Formatting
  - Python: PEP 8; use `ruff format` or `autopep8`.
  - You MUST hard-wrap every line (code, docstrings, comments)
    to a maximum of 79 characters.
  - You MUST break long function signatures/argument lists
    across multiple lines using a hanging indent and trailing
    commas.
  - You MUST prefer implicit parentheses for multi-line
    expressions and avoid backslashes.
  - You MUST keep imports grouped and avoid wildcard imports.
- Linting
  - Python: use `ruff`; project config is in `pyproject.toml`
  - Avoid wildcard imports; keep functions small and
    side-effect free.
- Naming & structure
  - `NormativeModel` is the top-level orchestrator; regression
    models (`BLR`, `HBR`) are pluggable backends.
  - Math building blocks (basis functions, scalers,
    likelihoods, priors) are composable and independent.
  - `NormData` is the central data container (xarray-based).
  - Use factory functions/classes where instantiation varies
    by config (e.g., `create_basis_function`,
    `regression_model/factory.py`).
  - When adding new functionality, try to create an abstract
    base class or interface to allow for future extensions.
  - Use `Output.print()`, `Output.warning()`, `Output.error()`
    for all user-facing messages. Define message templates in
    `Messages`, `Warnings`, or `Errors` classes in
    `util/output.py`.
- Commits & PRs
  - Use short subjects (≤50 chars), bodies wrapped at 72;
    prefer prefixes: `fix|enh|doc|cos|test`
  - Reference issues if applicable: `fix #123 - concise msg`
  - Keep PRs small and focused
- Docs
  - Add/update tutorial notebooks in `examples/` for new
    user-facing features
  - Keep `README.md` current with installation and usage info

# 7. Architecture and Design

- Components
  - **NormativeModel** (`normative_model.py`): orchestrates
    fitting, prediction, transfer, evaluation, and
    saving/loading. Wraps a `RegressionModel` and optional
    scalers.
  - **RegressionModel** (`regression_model/`): pluggable
    inference backend. `BLR` for fast analytical Bayesian
    regression; `HBR` for hierarchical Bayesian models via
    PyMC/nutpie.
  - **Math functions** (`math_functions/`): composable building
    blocks — basis functions, likelihoods, priors, scalers,
    warps.
  - **NormData** (`dataio/norm_data.py`): xarray-backed data
    container managing covariates, responses, batch effects,
    z-scores, centiles.
  - **Runner** (`util/runner.py`): manages parallel/cluster
    job submission (SLURM, Torque, local).
  - **Evaluator** (`util/evaluator.py`): computes evaluation
    metrics (RMSE, SMSE, EXPV, MSLL, BIC, etc.).
  - **CLI** (`normative.py`): command-line entry point
    exposing fit/predict/fit_predict.
- Data flow
  - User creates `NormData` (from DataFrame, CSV, or factory)
    → creates `NormativeModel` with chosen `RegressionModel`
    → calls `fit()` / `predict()` / `fit_predict()` →
    results stored back in `NormData` (z-scores, centiles,
    yhat, etc.).
  - For cluster jobs: `Runner` serializes model + data →
    submits batch jobs → monitors → collects results.
- Extension points
  - New regression models: subclass `RegressionModel`, register
    in `regression_model/factory.py`.
  - New basis functions: subclass `BasisFunction`, register in
    `create_basis_function`.
  - New likelihoods: create class in `likelihood.py`, wire into
    HBR.
  - New priors: add distribution to `PM_DISTMAP` in `prior.py`.
  - New scalers: subclass `Scaler` in `scaler.py`.
  - New evaluation metrics: add method to `Evaluator`.
  - New plotting: add function to `plotter.py`.

# 8. Deployment and Release Process

- Versioning
  - Version is defined in `pyproject.toml` under
    `[project].version`.
  - `CITATION.cff` is generated from `citation.cff.in` via
    `render_citation.py`.
- Publishing
  - Package is published to PyPI as `pcntoolkit`.
  - Install: `pip install pcntoolkit`
- Documentation
  - Hosted on ReadTheDocs; config in `.readthedocs.yaml`.
  - Sphinx builds from `doc/` directory.
- Branching
  - Default branch: `master`
  - Feature branches merged via PRs to the dev branch. Only when dev branch is stable, merge to master and tag for release.

# 9. Performance and Observability

- HBR sampling can be slow; nutpie is the default sampler for
  speed. Adjust `draws`, `tune`, `chains` for trade-off.
- BLR is fast (analytical); prefer for large-scale runs.
- `Runner` supports parallelization across response variables
  via batch jobs.
- Use `Output.set_show_messages(False)` to suppress verbose
  output in notebooks/scripts.
- No formal logging framework yet; `Output` class handles
  all messaging.

# 10. Critical Reminders

- Keep this file updated when architecture, deps, or CI change.
- Prefer deterministic, copy-pasteable commands; assume repo
  root.
- Do not commit test resources or large files (see `.gitignore`).
- All public API should be re-exported in `pcntoolkit/__init__.py`
  and listed in `__all__`.
- `NormData` is the single data container; do not bypass it
  with raw arrays in user-facing code.
- `Output`/`Messages`/`Warnings`/`Errors` in `util/output.py`
  is the single source of truth for all user-facing text.
- Regression models must be stateless after `fit()` — all
  learned state lives in the model object, not module globals.
- Use `cloudpickle` (not `pickle`) for serialization of models.

# 11. Type hints

- All new or modified Python functions/methods must annotate
  every parameter and the return type. Use `-> None` for
  procedures.
- Annotate module & class level variables (and dataclass fields)
  with explicit types; annotate important locals when
  readability benefits (skip trivial loop indices/counters).
- Prefer built-in generics (`list[str]`, `dict[str, Any]`,
  `tuple[int, ...]`) over `typing.List` style. Use `| None`
  (PEP 604) instead of `Optional[...]` unless clarity improves.
- Avoid `Any`; if unavoidable, add a short comment why. Narrow
  union types instead of broad `Any` (e.g., `int | float` not
  `Number`).
- Use `Literal` for fixed sets of string/int tokens when not
  already represented by an `Enum`. Prefer `Enum` for
  larger/shared symbolic sets.
- **MUST NOT** use capitalized versions (`Dict`, `List`,
  `Tuple`, `Set`, etc.) from the `typing` module. Always use
  lowercase built-in generic types (`dict`, `list`, `tuple`,
  `set`).

# 12. Docstrings

- Precise wording, clear reasoning, and straightforward
  argumentation.
- Use PEP 257 instructions for Python docstrings. One-liners
  are for really obvious cases. Multi-line docstrings consist
  of a summary line, followed by a blank line, followed by a
  more elaborate description.
- You MUST keep line length <= 79 characters for all docstrings
  and re-wrap as needed.
- For classes, also specify the attributes.
- Use NumPy-style `Parameters`, `Returns`, `Raises` sections in the docstrings

# 13. Comments

- Precise wording, clear reasoning, and straightforward
  argumentation.
- Add comments to every line of code, explaining what each
  line does.
- Do not edit already existing comments unless instructed to
  do so.
- You MUST keep line length <= 79 characters for all comments
  and re-wrap as needed.
- When editing existing code, do not write comments that
  compare the new code to previous approaches. All comments
  must be precise, focused on the current logic, and must not
  reference prior code or alternative implementations.

# 14. Pytest Unit Tests

- Use `pytest` as the test framework.
- Test files live in `test/` mirroring the package structure.
- Use descriptive test function names:
  `test_<id>_<function>_should_<expectation>_when_<condition>`.
  Example:
  `test_001_fit_should_returnError_when_dataIsEmpty`
- Structure tests with **Arrange–Act–Assert (AAA)**: setup →
  call method → assert results.
- Define reusable fixtures with `@pytest.fixture` in
  `test/fixtures/` or `test/conftest.py`.
- Each test should validate a **single behavior**; avoid
  combining unrelated assertions.
- Each test should contain only **one assert**. If multiple
  asserts are present, split them into separate tests.
- Assert expected state with clear comparisons; avoid
  over-fragmented assertions.
- Cover **edge cases**, including invalid inputs and boundary
  conditions.
- Use `@pytest.mark.parametrize` for input/output variations
  instead of duplicating test code.
- Keep test functions short and focused (≤30 lines).
- You must follow sections 6, 12, 13, and 14 for test code.
- Mark slow tests (e.g., HBR sampling) with
  `@pytest.mark.slow` so they can be skipped in quick runs.

# 15. Error handling

- Proactively identify and handle all possible edge cases.
- Use the `Errors` class in `util/output.py` for error message
  templates. Reuse existing templates; add new ones there if
  no suitable template exists.
- Use `Output.error()` to format error messages with context.
- Raise specific exceptions (`ValueError`, `FileNotFoundError`,
  `TypeError`, etc.) with formatted error messages.
- **Guard Clauses Pattern**: fail fast with early returns or
  raises for error conditions. Check what is NOT expected and
  handle immediately. Example:
  ```python
  if X is None:
      raise ValueError(
          Output.error(Errors.ERROR_HBRDATA_X_NOT_PROVIDED)
      )
  ```
- Use `Output.warning()` with templates from `Warnings` class
  for non-fatal issues.
- Never silently swallow exceptions; at minimum log them via
  `Output.warning()`.