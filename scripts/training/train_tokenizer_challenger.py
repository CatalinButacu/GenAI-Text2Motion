# python scripts/training/train_tokenizer_challenger.py -- --config configs/tokenizer/rvq_l8_1024.yaml --tokenizer rvq --epochs 500 --window 64 --batch_size 128 --ema_decay 0.99 --eval_every 25 --ckpt_name rvq_l8_1024_c1b.pt
from __future__ import annotations

import sys
from pathlib import Path

from text2motion.app.bootstrap import ApplicationBootstrap
from text2motion.app.cli import parse_request
from text2motion.app.cli_requests import TrainTokenizerRequest
from text2motion.app.training import run_train_tokenizer
from text2motion.motion.contracts import Split

CHALLENGER_SUFFIX = "_c1b.pt"


def challenger_problems(name: str | None, resume: bool, checkpoint_dir: Path) -> list[str]:
    if not name or not name.endswith(CHALLENGER_SUFFIX):
        return [
            f"--ckpt_name must end with {CHALLENGER_SUFFIX}, so a production checkpoint is never written"
        ]
    problems = []
    if resume:
        problems.append("a challenger starts fresh; --resume is not allowed")
    for path in (checkpoint_dir / name, checkpoint_dir / f"{Path(name).stem}_last.pt"):
        if path.exists():
            problems.append(f"{path} exists; a challenger never overwrites or continues a file")
    return problems


def main() -> None:
    request = parse_request(
        "train-tokenizer", [argument for argument in sys.argv[1:] if argument != "--"]
    )
    if not isinstance(request, TrainTokenizerRequest):
        raise TypeError("the train-tokenizer request did not parse")
    app = ApplicationBootstrap.from_config(request.config)
    problems = challenger_problems(
        request.ckpt_name, request.resume, app.checkpoint_directory("tokenizer")
    )
    if problems:
        raise SystemExit("refused: " + "; ".join(problems))
    run_train_tokenizer(app, request, final_split=Split.VALIDATION)


if __name__ == "__main__":
    main()
