Contributing
============

First off, thanks for considering contributing to PCNtoolkit! 🎉👍

The following is a set of guidelines for contributing to PCNtoolkit. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

Reporting Bugs
---------------------

This section guides you through submitting a bug report. Following these guidelines helps maintainers and the community understand your report, reproduce the behavior, and find related reports.

Before Submitting A Bug Report
""""""""""""""""""""""""""""""

* Ensure the bug is not already reported by searching on GitHub under `Issues <https://github.com/amarquand/PCNtoolkit/issues>`_

How Do I Submit A Good Bug Report?
""""""""""""""""""""""""""""""""""

Bugs are tracked as `GitHub issues <https://github.com/amarquand/PCNtoolkit/issues>`_. Create an issue and provide the following information:

* Use a clear and descriptive title for the issue to identify the problem
* Describe the exact steps which reproduce the problem in as much detail as possible
* Provide specific examples to demonstrate the steps. Include links to files or GitHub projects, or copy/pasteable snippets

Suggesting New Features
------------------------

This section guides you through submitting an enhancement suggestion, including completely new features and minor improvements to existing functionality.

How Do I Submit A Good Enhancement Suggestion?
""""""""""""""""""""""""""""""""""""""""""""""

Enhancement suggestions are tracked as `GitHub issues <https://github.com/amarquand/PCNtoolkit/issues>`_. Create an issue and provide the following information:

* Use a clear and descriptive title for the issue to identify the suggestion
* Provide a step-by-step description of the suggested enhancement in as much detail as possible
* Provide specific examples to demonstrate the steps

Setting up your local development environment
-----------------

You are always welcome to contribute code yourself. PCNtoolkit runs on Linux, Mac, or Windows with WSL. To contribute to PCNtoolkit you can follow the steps below to set your local development environment:

.. note::
    To use PCNtoolkit you need to have installed `Git <https://git-scm.com/downloads>`_ and a Python version higher than 3.9 and lower than 3.13. Also, we recommend using `Anaconda <https://www.anaconda.com/download>`_ to manage Python.

1. **Fork the repository** - Forking is the process of creating your own personal copy of the project on GitHub. The fork lives under your GitHub account and lets you experiment, make changes, or contribute improvements without affecting the original project. See `GitHub’s guide to forking <https://docs.github.com/en/get-started/quickstart/fork-a-repo>`_.

2. **Clone your fork** - Cloning is the process of downloading your GitHub copy of the project onto your computer, so you can make changes to the code locally:

   .. code-block:: bash

       git clone https://github.com/your-username/PCNtoolkit.git

3. **Create environment** - A Python environment is an isolated workspace that contains its own Python installation and libraries. This prevents conflicts with other Python projects on your system. Here, we create an environment called ``.ptk-dev`` and then activate it.

   .. code-block:: bash

       conda create -n .ptk-dev
       conda activate .ptk-dev

4. **Install the dependencies** 

   .. code-block:: bash

       pip install -e ".[dev]"

   The idea behind ``pip install -e`` is to make it easy to work on the code while you are developing it. Instead of treating the project as a finished product, Python treats it as “live” code, so you can edit it and test your changes right away. Also, ``".[dev]"`` lets you install extra dependencies that are useful when developing the project (for example for testing, or documentation).

.. note::
    **Alternative quicker option:** To simplify and automate commands that are frequently used during development, we use `GNU Make <https://www.gnu.org/software/make/>`_. Common development tasks are defined as short scripts in the ``Makefile``. For example, steps 3 and 4 can be done with a single command: ``make dev-setup``. After it, you should activate the environment with ``conda activate .ptk-dev``.

Congrats! You have now set up your development environment. 

To contribute, create a new branch based on the ``dev`` branch and open your pull request. Your pull request should be approved by at least one developer before merging. The ``dev`` branch is where we add changes for the next release and once we are ready to release to the public we move all our code to the ``master`` branch.

Running Tests
-------------

.. code-block:: bash

    pytest test/

Building the Website
--------------------

Our website lives in the ``doc/`` folder and is built with Sphinx. When you make changes to existing documentation or add new documentation, you should build the website locally to check how everything looks. To do that:

1. Go to the ``doc/`` folder:

   .. code-block:: bash

       cd doc

2. Install the documentation dependencies:

   .. code-block:: bash

       pip install -r requirements.txt

3. Build the website and keep it up to date: every time you save a change, the page reloads in your browser.

   .. code-block:: bash

       make livehtml

Code style
-----------

Git Commit Messages
"""""""""""""""""""

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

Python code style
"""""""""""""""""""

All Python code must adhere to `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_. We recommend using the ``autopep8`` python package to help you automatically format your code based on the PEP8 standard. Please see the `autopep8 documentation <https://github.com/hhatto/autopep8>`_ for more details.
