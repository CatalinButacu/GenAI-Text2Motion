.PHONY: help demo test smoke lint verify-data pin docker-demo train-rvq train-ssm

PY ?= python

help:
	@echo "make demo        -- run main.py on the default prompts (defense demo)"
	@echo "make test        -- pytest (skip slow/integration)"
	@echo "make smoke       -- pytest tests/test_smoke.py -m slow  (requires checkpoints)"
	@echo "make verify-data -- refresh data fingerprint baseline"
	@echo "make pin         -- refresh requirements.lock.txt"
	@echo "make train-rvq   -- train RVQ tokenizer (prereq for train-ssm)"
	@echo "make train-ssm   -- train MotionSSM on AMASS"
	@echo "make docker-demo -- build + run the demo container"

demo:
	$(PY) main.py "a person walks forward"              --name walk        --duration 4
	$(PY) main.py "a person walks and kicks a ball"     --name walk_kick   --duration 8
	$(PY) main.py "a person jumps"                      --name jump        --duration 3

test:
	$(PY) -m pytest -m "not slow and not gpu"

# pytest-xdist is available but `-n auto` is currently slower for us than
# sequential because each worker re-imports torch+spacy (heavy cold start)
# and most tests finish in under 100ms. Re-evaluate this default when the
# suite grows past ~500 tests or a single slow test dominates wallclock.
test-parallel:
	$(PY) -m pytest -m "not slow and not gpu" -n auto

smoke:
	$(PY) -m pytest tests/test_smoke.py::test_pipeline_runs -v

verify-data:
	$(PY) scripts/verify_data.py

pin:
	$(PY) -m pip freeze > requirements.lock.txt

train-rvq:
	$(PY) scripts/training/train_rvq_tokenizer.py --data-dir data/AMASS

train-ssm:
	$(PY) scripts/training/train_motion_ssm.py --data-source amass --use-sbert --bidirectional --use-film

docker-demo:
	docker build -t motion-demo -f Dockerfile .
	docker run --rm -v $(PWD)/outputs:/app/outputs -v $(PWD)/checkpoints:/app/checkpoints:ro \
	           -v $(PWD)/data/arctic/unpack/models:/app/data/arctic/unpack/models:ro \
	           motion-demo "a person walks forward"
