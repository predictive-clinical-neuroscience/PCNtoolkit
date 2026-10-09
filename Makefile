ENV_NAME = ptk-dev
PYTHON_VERSION = 3.12

.PHONY: dev-setup venv

dev-setup:
	conda create -n $(ENV_NAME) python=$(PYTHON_VERSION) -y
	conda run -n $(ENV_NAME) pip install -e ".[dev]"

venv:
	uv venv --clear --python $(PYTHON_VERSION) .venv
	uv pip install --python .venv -e ".[dev]"
