# make help
SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command
export PYTHONPATH := src
export PYTORCH_CUDA_ALLOC_CONF := expandable_segments:True

PY := .\.venv\Scripts\python.exe
CLI := $(PY) -u -m text2motion.app.cli
PS := powershell -NoProfile -ExecutionPolicy Bypass -File

B ?= mamba
E ?= 60
NAME ?= dev
CFG ?= configs/generator/gen_pilot_fsq8x1024.yaml
TOK ?= artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt
GATE ?= artifacts/outputs/gates/$(B)_34m_overfit.json
CKPT ?= artifacts/checkpoints/generator/generator_$(B)_bs8_e60.pt
LEVEL ?= INFO

.DEFAULT_GOAL := help
.PHONY: help install test lint fix types check clean gpu \
        prep corpus tok gate pre train twins ext100 dyn rvq resume \
        val eval eval-dyn bench serve studio health \
        pdf figs runs runs-apply lic graph last

help:
	@echo "Variables: B=mamba|transformer  E=epochs  NAME=run name  CFG=config  CKPT=checkpoint  LEVEL=DEBUG|INFO"
	@echo ""
	@echo "SETUP     install    sync the environment from uv.lock"
	@echo "          check      lint and tests (the green gate)"
	@echo "          test lint fix types clean gpu"
	@echo "DATA      prep       AMASS and HumanML3D to 263 features"
	@echo "          corpus     build the unlabeled AMASS pretraining corpus"
	@echo "          tok        encode the corpus into tokens"
	@echo "TRAIN     gate       overfit one batch (required before train)       make gate B=mamba"
	@echo "          pre        unconditional pretraining, one backbone         make pre B=mamba"
	@echo "          train      one short unguarded run                         make train B=mamba E=6 NAME=try1"
	@echo "          twins      guarded 60-epoch run of the two twins (days)"
	@echo "          ext100     guarded extension of the two twins to 100 epochs"
	@echo "          dyn        guarded dynamics fine-tune of the two twins     make dyn E=72"
	@echo "          rvq        guarded RVQ-token ablation"
	@echo "          resume     resume the 60-epoch mamba run"
	@echo "EVALUATE  val        validation metrics for one checkpoint           make val B=mamba CKPT=..."
	@echo "          eval       CITABLE test evaluation of the two twins (run one time)"
	@echo "          eval-dyn   test evaluation and naturalness of the dynamics checkpoints"
	@echo "          bench      streaming state and latency benchmark"
	@echo "RUN       studio     interactive viewer                              make studio B=mamba"
	@echo "          serve health"
	@echo "THESIS    pdf        build doc/paper/thesis.pdf"
	@echo "          figs       regenerate the figures"
	@echo "UPKEEP    runs       show which run folders continue which (dry run)"
	@echo "          runs-apply merge them"
	@echo "          lic        regenerate THIRD_PARTY_LICENSES.md"
	@echo "          last       last 5 metric rows of the newest run"
	@echo "          graph      update the code graph"

install:
	uv sync --extra dev --extra viewer

test:
	$(PY) -m pytest -q -m "not slow and not gpu"

lint:
	$(PY) -m ruff check .

fix:
	$(PY) -m ruff check --fix . ; $(PY) -m ruff format .

types:
	.\.venv\Scripts\pyright.exe --pythonpath $(PY)

check: lint test

clean:
	Get-ChildItem src,tests,scripts -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force -Confirm:$$false
	Remove-Item .pytest_cache,.ruff_cache -Recurse -Force -Confirm:$$false -ErrorAction SilentlyContinue

gpu:
	nvidia-smi

prep:
	$(CLI) --log_level $(LEVEL) prepare --config configs/default.yaml --stage all

corpus:
	$(CLI) --log_level $(LEVEL) prepare --config configs/default.yaml --corpus

tok:
	$(CLI) --log_level $(LEVEL) tokenize --config $(CFG) --tokenizer_ckpt $(TOK) --token_pack data/amass_tokens_fsq8x1024.npz

gate:
	$(CLI) --log_level $(LEVEL) sanity-overfit --config $(CFG) --backbone $(B) --tokenizer_ckpt $(TOK) --overfit_gate $(GATE)

pre:
	$(CLI) --log_level $(LEVEL) pretrain --config $(CFG) --backbone $(B) --epochs 20 --batch_size 8 --resume

train:
	$(CLI) --log_level $(LEVEL) train-generator --config $(CFG) --backbone $(B) --tokenizer_ckpt $(TOK) --overfit_gate $(GATE) --epochs $(E) --batch_size 8 --cfg_scale 6.0 --eval_every 6 --ckpt_name generator_$(B)_$(NAME).pt

twins:
	$(PS) scripts/training/run_matched_twin_60ep_local.ps1

ext100:
	$(PS) scripts/training/extend_twin_to_100ep_local.ps1

dyn:
	$(PS) scripts/training/finetune_dynamics_twins_local.ps1 -Epochs $(E)

rvq:
	$(PS) scripts/training/run_rvq_ablation_local.ps1

resume:
	$(PS) scripts/training/resume_mamba60.ps1

val:
	$(CLI) --log_level $(LEVEL) evaluate --config $(CFG) --backbone $(B) --split val --ckpt $(CKPT) --tokenizer_ckpt $(TOK) --cfg_scale 6.0 --max_clips 600

eval:
	$(PS) scripts/evaluation/twin_test_eval.ps1 -NoWait

eval-dyn:
	$(PS) scripts/evaluation/dynamics_ablation_eval.ps1

bench:
	$(CLI) --log_level $(LEVEL) benchmark --config configs/generator/final100m_fsq8x1024.yaml

serve:
	$(CLI) --log_level $(LEVEL) serve --config $(CFG) --backbone $(B) --ckpt $(CKPT) --tokenizer_ckpt $(TOK)

studio:
	$(CLI) --log_level $(LEVEL) studio --config $(CFG) --backbone $(B) --ckpt $(CKPT) --tokenizer_ckpt $(TOK)

health:
	$(CLI) health

pdf:
	Push-Location doc/paper; xelatex -interaction=nonstopmode -output-directory=build thesis.tex | Out-Null; biber --input-directory build --output-directory build thesis | Out-Null; xelatex -interaction=nonstopmode -output-directory=build thesis.tex | Out-Null; xelatex -interaction=nonstopmode -output-directory=build thesis.tex | Select-String 'Output written'; Copy-Item build/thesis.pdf thesis.pdf -Force; Pop-Location

figs:
	$(PY) scripts/figures/make_figures.py

runs:
	$(PY) scripts/runs/merge_runs.py

runs-apply:
	$(PY) scripts/runs/merge_runs.py --apply

lic:
	$(PY) scripts/licenses/collect_licenses.py

last:
	Get-ChildItem artifacts/logs/train -Directory | Where-Object { Test-Path (Join-Path $$_.FullName 'metrics.jsonl') } | Sort-Object LastWriteTime | Select-Object -Last 1 | ForEach-Object { $$_.Name; Get-Content (Join-Path $$_.FullName 'metrics.jsonl') -Tail 5 }

graph:
	graphify update .
