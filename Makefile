ENV_NAME = ptk-dev
PYTHON_VERSION = 3.13

.PHONY: dev-setup uv-dev-setup

dev-setup:
	conda create -n $(ENV_NAME) python=$(PYTHON_VERSION) -y
	conda run -n $(ENV_NAME) pip install -e ".[dev]"

uv-dev-setup:
	uv venv --clear --python $(PYTHON_VERSION) .$(ENV_NAME)
	uv pip install --python .$(ENV_NAME) -e ".[dev]"
