# Evaluation harness targets for the frozen benchmark.
# Prefer an active virtualenv if one exists, otherwise fall back to python3.
VENV := $(wildcard ../../.venv/bin/python)
PYTHON ?= $(if $(VENV),$(VENV),python3)

.PHONY: help benchmark test

help:
	@echo "make benchmark  - regenerate Table 5 from results/preds/"
	@echo "make test       - run the scoring harness unit tests"

benchmark:
	PYTHON=$(PYTHON) ./bin/run_table5.sh

test:
	$(PYTHON) tests/test_score.py
