PY := uv run python
CLI := $(PY) -u -m text2motion.app.cli
PWSH := powershell -NoProfile -ExecutionPolicy Bypass -File
CONFIG ?= configs/default.yaml

.DEFAULT_GOAL := help
.PHONY: help install test lint fix types check prepare tokenize overfit \
        train-tokenizer train pretrain train-local eval bench serve studio health graph clean

help:  ## show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-16s %s\n", $$1, $$2}'

install:  ## sync the venv from uv.lock
	uv sync --extra dev

test:  ## run the test suite
	$(PY) -m pytest -q

lint:  ## ruff check
	$(PY) -m ruff check .

fix:  ## ruff check --fix
	$(PY) -m ruff check --fix .

types:  ## pyright type check (project-wide; backlog, not yet green)
	uv run pyright

check: lint test  ## the green gate: lint + tests

prepare:  ## AMASS -> HumanML3D-263
	$(CLI) prepare --config $(CONFIG)

tokenize:  ## encode a feature corpus into tokens
	$(CLI) tokenize --config $(CONFIG)

overfit:  ## MANDATORY pre-flight: overfit one real batch
	$(CLI) sanity-overfit --config $(CONFIG)

train-tokenizer: overfit  ## train the tokenizer (SHORT runs only, unguarded)
	$(CLI) train-tokenizer --config $(CONFIG)

train: overfit  ## train the generator (SHORT runs only, unguarded)
	$(CLI) train-generator --config $(CONFIG)

pretrain:  ## unconditional token-corpus pretraining (SHORT runs only, unguarded)
	$(CLI) pretrain --config $(CONFIG)

train-local:  ## the guarded multi-day local pipeline (watchdog + hour budget)
	$(PWSH) scripts/training/run_3day_local.ps1

eval:  ## citable 20-rep evaluation on the test split
	$(CLI) evaluate --config $(CONFIG)

bench:  ## streaming latency / state-size benchmark
	$(CLI) benchmark --config $(CONFIG)

serve:  ## local streaming generation service
	$(CLI) serve --config $(CONFIG)

studio:  ## interactive aitviewer studio
	$(CLI) studio --config $(CONFIG)

health:  ## check the streaming service readiness
	$(CLI) health --config $(CONFIG)

graph:  ## re-extract code files into the knowledge graph
	graphify update .

clean:  ## remove caches only; never touches data/, outputs/ or checkpoints/
	$(PY) -c "import pathlib,shutil; [shutil.rmtree(q, ignore_errors=True) for p in ('src','tests','scripts') for q in pathlib.Path(p).rglob('__pycache__')]"
	$(PY) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('.pytest_cache','.ruff_cache','.ropeproject')]"
