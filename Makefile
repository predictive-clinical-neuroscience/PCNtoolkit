ENV_NAME = .ptk-dev

.PHONY: dev-setup

dev-setup:
	conda create -n $(ENV_NAME) -y
	conda run -n $(ENV_NAME) pip install -e ".[dev]"
