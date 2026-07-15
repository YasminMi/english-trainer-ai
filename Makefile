.PHONY: install test run-cli run-tui check

install:
	pip install -r requirements-dev.txt

test:
	python -m pytest -q

run-cli:
	python main.py --cli

run-tui:
	python main.py

check:
	python -m compileall english_trainer
