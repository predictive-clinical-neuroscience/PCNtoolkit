# Predictive Clinical Neuroscience Toolkit

[![PyPI version](https://img.shields.io/pypi/v/pcntoolkit.svg)](https://pypi.org/project/pcntoolkit/)
[![Python versions](https://img.shields.io/pypi/pyversions/pcntoolkit.svg)](https://pypi.org/project/pcntoolkit/)
[![Documentation Status](https://readthedocs.org/projects/pcntoolkit/badge/?version=latest)](https://pcntoolkit.readthedocs.io/en/latest/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7498917.svg)](https://doi.org/10.5281/zenodo.7498917)

Predictive Clinical Neuroscience software toolkit (formerly nispat). 

A Python package for normative modelling, spatial statistics and pattern recognition.

# IMPORTANT 
## Deprecation warning

This is PCNtoolkit version 1.X.X, released originally in June 2025. Any scripts, models, and results created with version 0.X.X are **not compatible** with this and future versions of the toolkit. 

To use the models created with versions 0.35 and earlier, please install the appropriate version using `pip install pcntoolkit==0.35`, or replace 0.35 with your desired version. The old version of the toolbox is also still available on [GitHub](https://github.com/amarquand/PCNtoolkit/tree/v0.35).

## Installation

```bash
pip install pcntoolkit
```

## Documentation

See the [documentation](https://pcntoolkit.readthedocs.io/en/latest/) for more details.

Documentation for the earlier version of the toolbox is available [here](https://pcntoolkit.readthedocs.io/en/v0.35/)

## Example usage

```python
from pcntoolkit import {load_fcon, BLR, NormativeModel}

fcon1000 = load_fcon()

train, test = fcon1000.train_test_split()

# Create a BLR model with heteroskedastic noise
model = NormativeModel(BLR(heteroskedastic=True), 
                       inscaler='standardize', 
                       outscaler='standardize')

model.fit_predict(train, test)
```

## Getting help

- **Usage questions:** ask on [NeuroStars](https://neurostars.org/tags/pcntoolkit) using the tag `pcntoolkit`.
- **Bugs and feature requests:** [open an issue](https://github.com/predictive-clinical-neuroscience/PCNtoolkit/issues).

When reporting a bug, please include your Python and PCNtoolkit versions, a minimal reproducible example, and the full traceback.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](https://github.com/predictive-clinical-neuroscience/PCNtoolkit/blob/dev/CONTRIBUTING.md) for how to set up a development environment, run the tests, and format your code.

Our [GitHub Wiki](https://github.com/predictive-clinical-neuroscience/PCNtoolkit/wiki) has more depth: a **Contributors guide** (branching and PR workflow, adding tutorials to the website, software architecture) and a **Maintainers guide** (making a release, migrating saved models).

## Citing PCNtoolkit

If you use PCNtoolkit in your research, please cite the software and the accompanying paper.

**Software** — [10.5281/zenodo.7498917](https://doi.org/10.5281/zenodo.7498917). This DOI always resolves to the latest release. Full machine-readable metadata is in [CITATION.cff](https://github.com/predictive-clinical-neuroscience/PCNtoolkit/blob/dev/CITATION.cff).

**Paper (version 1.x)** — de Boer, A. A. A., Bayer, J. M. M., Fraza, C., et al. (2026). *Protocol Update: The Normative Modelling Paradigm for Computational Psychiatry.* bioRxiv. [10.64898/2026.02.17.706268](https://doi.org/10.64898/2026.02.17.706268)

**Paper (version 0.x)** — Rutherford, S., Kia, S. M., Wolfers, T., et al. (2022). *The Normative Modeling Framework for Computational Psychiatry.* Nature Protocols. [10.1038/s41596-022-00696-5](https://doi.org/10.1038/s41596-022-00696-5)

## License

PCNtoolkit is released under the [GPL-3.0-only](https://github.com/predictive-clinical-neuroscience/PCNtoolkit/blob/dev/LICENSE) license.

