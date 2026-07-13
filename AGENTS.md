# AGENTS.md

This file provides guidance to AI coding agents working with this repository.

## What is PCNtoolkit

PCNtoolkit is an open-source Python package for Normative Modelling of neuroimaging data. Our target users are neuroscience researchers.

## Contribution guidelines
ALWAYS follow our contribution guidelines. Fetch and read these before contributing:
- Website: https://pcntoolkit.readthedocs.io/en/stable/pages/contributing.html
- GitHub wiki: https://github.com/predictive-clinical-neuroscience/PCNtoolkit/wiki

## GitHub workflow

- `master` is the release branch, 
- `dev` is the active development branch. 
- Every contributor MUST work on a feature branch based on `dev` and the branch MUST follow the name conventions: <github-username>/<short-feature-description>. 
- Pull requests must point to `dev`. 
- Every Pull Request MUST define if AI was used to generate the code

## Software architecture

We try to follow as much as possible the scikit-learn API.

### Scikit-learn API

**Normative Modelling**

- `NormativeModel(BLR(...), inscaler=..., outscaler=...).`: `NormativeModel` is a meta-estimator wrapping a regression estimator `BLR` or `HBR`, mirroring the sklearn `GridSearchCV(estimator)` pattern.
- Meta-estimator and estimator methods: `fit()` / `predict()` / `fit_predict`.
- Regression estimators expose `forward` / `backward` / `elemwise_logp`,
- Federated learning meta-estimator and estimator methods: `transfer`/ `extend` / `merge`.

**Longitudinal Normative Modelling**

- Longitudinal model method: `score()`

### Non-scikit-learn API

What is *not* borrowed from scikit-learn:

- Data is `NormData`, not raw `X, y` arrays. It is xarray, so data carries named
  dimensions (observations, response_vars, covariates, batch_effect_dims, ...) rather than
  being a flat 2D matrix.
- No trailing-underscore learned-attribute convention.
- Models are saved and loaded from readable json files with `to_dict` / `from_dict`, not raw pickling.

For more detailed software architecture fetch and read the [architecture description from our github wiki](https://github.com/predictive-clinical-neuroscience/PCNtoolkit/wiki/Software-architecture#architecture-description)

## CLI tooling

You MUST use these CLI tools for the corresponding tasks (check `--help` for exact commands):

- **conda** -  virtual environment setup (fetch and follow the guidelines from https://pcntoolkit.readthedocs.io/en/stable/pages/contributing.html)
- **ruff** - lint and format.
- **pytest** - tests, under `test/`
- **gh** - branches and PRs
- **make** - automate dev tasks. The most useful task is building the website with `cd doc && make livehtml`) (for more tasks see `Makefile` and `doc/Makefile`)

## Developing in Windows
PCNtoolkit officially works in Linux and Mac. However, it is possible to contribute from a Windows laptop without WSL. 

For Windows you have to manually install the g++ compiler (to compile C/C++ extensions when you run a HBR model), as it is not installed by default in Windows. To install it run:

`conda install -c conda-forge m2w64-toolchain libpython`

Windows does not come with Make pre-installed. You can use Make via Git Bash. To find instructions about that fetch and read https://github.com/predictive-clinical-neuroscience/PCNtoolkit/wiki/PCNtoolkit-from-Windows#2-installing-make-on-windows.

## CI/CD

## Code Style and Standards

### Docstrings
### Comments
- Comment the non-obvious lines only

### Type hints
- All new or modified Python functions/methods must annotate
  every parameter and the return type.