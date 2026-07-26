.DEFAULT_GOAL := tasks

PYTHON ?= python3
DATASETS ?= all
RAW_DIR ?= data/raw
OUTPUT ?= data/tasks.csv
VARIANTS ?= full,oracle

.PHONY: install tasks test

install:
	$(PYTHON) -m pip install -r requirements.txt

tasks:
	$(PYTHON) build_csv.py \
		--datasets "$(DATASETS)" \
		--raw-dir "$(RAW_DIR)" \
		--output "$(OUTPUT)" \
		--variants "$(VARIANTS)"

test:
	$(PYTHON) -m pytest -q
