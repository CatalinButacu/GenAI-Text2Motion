"""Fine-tune GPT-2-small as the agent's action-decomposition planner.

Decision #14 (GPT-2-small, MIT, 124M params) + decision #17a (synthetic
HumanML3D-derived dataset). The planner takes a free-form natural-language
instruction and emits a JSON list of (action, until) tuples conforming to
the runtime grammar (see src/modules/agent/conditions.py).

Prompt format (kept deliberately simple -- GPT-2 has no chat template):

    Instruction: walk forward then sit down
    Actions: [{"action": "walk forward", "until": "duration(40)"}, ...]

Loss is masked on the prompt portion so the model only learns to complete
after "Actions: ", not to memorise instruction surface forms.

Run from repo root:
    python scripts/training/train_planner_lm.py \
        --dataset data/planner \
        --output checkpoints/planner_lm \
        --epochs 3 \
        --batch-size 16

VRAM footprint: GPT-2-small + AdamW + activations at batch=16 fits in
roughly 1.5 GB. Full-SFT (no LoRA) for simplicity at this size.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)

PROMPT_TEMPLATE = "Instruction: {instruction}\nActions: "


@dataclass
class PlannerExample:
    """One (prompt, completion) pair for SFT.

    Loss is computed only on the completion -- the prompt is masked out
    so the model doesn't waste capacity learning to predict the surface
    form of the instruction we just gave it.
    """

    prompt: str
    completion: str


class PlannerDataset(Dataset):
    """JSONL -> tokenised prompt+completion with a label mask.

    Loading is eager (the dataset is ~6000 lines, well under 5 MB) so we
    don't add a streaming-shuffle wrinkle on top of the SFT loop.
    """

    def __init__(self, jsonl_path: Path, tokenizer, max_length: int = 256) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: list[PlannerExample] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                prompt = PROMPT_TEMPLATE.format(instruction=obj["instruction"])
                completion = json.dumps(obj["actions"], separators=(", ", ": "))
                self.examples.append(PlannerExample(prompt=prompt, completion=completion))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        # Tokenise prompt and completion separately so we can locate the
        # boundary precisely for the label mask. Add an explicit EOS so the
        # model learns where to stop.
        prompt_ids = self.tokenizer.encode(ex.prompt, add_special_tokens=False)
        completion_ids = self.tokenizer.encode(ex.completion + self.tokenizer.eos_token,
                                                add_special_tokens=False)
        input_ids = (prompt_ids + completion_ids)[: self.max_length]
        # -100 is the conventional "ignore" index for the cross-entropy loss
        labels = ([-100] * len(prompt_ids) + completion_ids)[: self.max_length]

        # Pad / mask to max_length
        pad_id = self.tokenizer.eos_token_id  # GPT-2 has no PAD, reuse EOS
        attention_mask = [1] * len(input_ids) + [0] * (self.max_length - len(input_ids))
        input_ids = input_ids + [pad_id] * (self.max_length - len(input_ids))
        labels = labels + [-100] * (self.max_length - len(labels))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    n_steps = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            total_loss += out.loss.item()
            n_steps += 1

    return total_loss / max(n_steps, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/planner",
                        help="Directory containing train.jsonl + val.jsonl")
    parser.add_argument("--output", default="checkpoints/planner_lm",
                        help="Where to save the fine-tuned model + tokenizer")
    parser.add_argument("--base-model", default="gpt2",
                        help="HuggingFace model id; gpt2 = GPT-2-small 124M")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-train", type=int, default=None,
                        help="If set, truncate train.jsonl to first N examples (fast smoke)")
    parser.add_argument("--max-val", type=int, default=None,
                        help="If set, truncate val.jsonl to first N examples (fast smoke)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda"
        else "cpu",
    )
    log.info("Device: %s", device)

    log.info("Loading base model %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model params: %.1fM", n_params / 1e6)
    ds_dir = Path(args.dataset)
    train_ds = PlannerDataset(ds_dir / "train.jsonl", tokenizer, args.max_length)
    val_ds = PlannerDataset(ds_dir / "val.jsonl", tokenizer, args.max_length)

    if args.max_train is not None:
        train_ds.examples = train_ds.examples[: args.max_train]

    if args.max_val is not None:
        val_ds.examples = val_ds.examples[: args.max_val]
    log.info("Train: %d examples  Val: %d examples", len(train_ds), len(val_ds))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    total_steps = len(train_loader) * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = OneCycleLR(
        optimizer, max_lr=args.learning_rate, total_steps=total_steps,
        pct_start=0.1, anneal_strategy="cos",
    )
    log.info("Starting fine-tune: %d epochs x %d steps = %d total",
             args.epochs, len(train_loader), total_steps)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = math.inf

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0

        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            n += 1

            if step % args.log_every == 0:
                log.info("epoch=%d step=%d/%d loss=%.4f lr=%.2e",
                         epoch, step, len(train_loader),
                         running / n, scheduler.get_last_lr()[0])
                running = 0.0
                n = 0
        val_loss = evaluate(model, val_loader, device)
        log.info("epoch=%d val_loss=%.4f", epoch, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            log.info("Saved best model to %s (val_loss=%.4f)", out_dir, val_loss)
    log.info("Done. Best val_loss=%.4f. Model + tokenizer in %s",
             best_val_loss, out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
