.DEFAULT_GOAL := help
PYTHON ?= python

.PHONY: help install lint format typecheck test cov check run run-cli clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Install the package in editable mode with dev tooling
	$(PYTHON) -m pip install -e ".[all]" --group dev
	pre-commit install

lint: ## Run ruff lint and format checks
	ruff check .
	ruff format --check .

format: ## Apply ruff fixes and formatting
	ruff check --fix .
	ruff format .

typecheck: ## Run mypy
	mypy

test: ## Run the test suite
	$(PYTHON) -m pytest -q

cov: ## Run tests with coverage
	$(PYTHON) -m pytest -q --cov --cov-report=term-missing

check: lint typecheck test ## Run every quality gate

run: ## Start the Textual TUI
	english-trainer

run-cli: ## Start the plain terminal interface
	english-trainer --cli

clean: ## Remove caches and build artifacts
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
