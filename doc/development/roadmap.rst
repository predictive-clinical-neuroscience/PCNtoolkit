Roadmap
=======

This page describes the future goals of PCNtoolkit.

1. Features & fixes
-------------------

**Extend longitudinal normative modelling**

* Version 1.4.0 added the z-diff score (`Bučková et al., 2025 <https://doi.org/10.7554/eLife.95823>`_), and the velocity models (z-gain score, thrivelines; `Bayer et al., 2026 (preprint) <https://arxiv.org/abs/2601.07591>`_).
* Forecast a subject's future z-score from their full visit history (conditional forecasting).
* Compute z-gain from all of a subject's visits, not only the last two.
* Compute your own correlation matrix.

**Improve federated learning**

* Make model transfer more robust and flexible.
* Deploy in clinical settings involving many hospitals.

**Expand normative models**

* Add new likelihoods for HBR to model discrete, zero-inflated, or other difficult distributions.
* Make HBR faster (e.g. explore an approximate HBR solution instead of sampling).

**Migrate PCNportal to the newest version of PCNtoolkit**

**Expand website documentation**

* Better explain difficult topics (teaching material on, e.g., federated learning and data handling with xarrays).
* Document federated learning with pretrained models, and provide a better way of documenting and sharing our models so they are findable via Google search.

**Explore agentic AI**

* Provide reusable skills for users that use agentic AI. In these skills normative modelling and PCNtoolkit will be explained.

**Data I/O**

* Make saving and loading data more flexible: save faster and in less space, and load selected data rather than the full dataset.

2. Software maintenance
-----------------------

**Testing**

* Find the test coverage of the whole codebase, unify test fixtures around shared data (e.g. create synthetic data in one place and reuse it across all other tests), and improve test coverage with more tests.

**Dependency management**

* Handle dependencies better so that future versions of PCNtoolkit don't break when our dependencies update.

**Track usage**

* Understand how our software is used and produce usage statistics.
