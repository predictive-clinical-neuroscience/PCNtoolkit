Roadmap
=======

This page describes the future goals of PCNtoolkit

1. Features & fixes
-------------------

**Add longitudinal normative modelling**

Normative models were originally cross-sectional: one snapshot, deviation scores at a single point. Longitudinal data carry clinical information that a single cross-sectional datapoint misses. For this reason we want to implement the longitudinal models by `Bučková et al., 2025 <https://elifesciences.org/reviewed-preprints/95823>`_ and `Bayer et al., 2026 (preprint) <https://arxiv.org/abs/2601.07591>`_

**Improve federated learning**

* Make model transfer more robust and flexible
* Deploy in clinical settings involving many hospitals.

**Expand normative models**

* Add new likelihoods for HBR to model discrete, zero-inflated, or other difficult distributions.
* Make HBR faster (e.g. explore an approximate HBR solution instead of sampling).

**Migrate PCNportal to the newest version of PCNtoolkit**

**Redesign website**

* Make it more modular, extendable and improve the aesthetic.
* Allow for users to download the tutorials as ipynbs.

**Expand website documentation**

* Cover the new longitudinal modelling,
* Better explain difficult topics (teaching material on, e.g., federated learning and data handling with xarrays).
* Document federated learning with pretrained models, and provide a better way of documenting and sharing our models so they are findable via Google search.

**Explore agentic AI**

Provide reusable skills for users that use agentic AI. In these skills normative modelling and PCNtoolkit will be explained.

**Data i/o**

Make saving and loading data more flexible: save faster and in less space, and load selected data rather than the full dataset.

2. Software maintenance
-----------------------

**Testing**

Find the test coverage of the whole codebase, unify test fixtures around shared data (e.g. create synthetic data in one place and reuse it across all other tests), and improve test coverage with more tests.

**Dependency management**

* Handle dependencies better so that future versions of PCNtoolkit don't break when our dependencies update.
* Add Python 3.13 support

**Track usage**

Understand how our software is used and produce usage statistics.
