"""ActionPlanner: fine-tuned GPT-2-small wrapper that decomposes natural-language
instructions into JSON action plans.

Loads a fine-tuned checkpoint produced by ``scripts/training/train_planner_lm.py``,
generates with grammar-enforced sampling (via ``outlines`` when installed),
parses the output, and returns a list of :class:`PlannedAction` -- each one
carrying a free-form ``action`` text the MotionSSM will condition on, plus
a parsed :class:`ParsedCondition` for the runtime termination check.

The planner is the single bridge between user input and the motion model;
if it emits a malformed plan we reject it and surface the error rather than
silently dropping the action. The runtime trusts only what the parser
accepts, so unparseable LM output never reaches the motion stack.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .conditions import ParsedCondition, parse_condition

log = logging.getLogger(__name__)

PROMPT_TEMPLATE = "Instruction: {instruction}\nActions: "


@dataclass
class PlannedAction:
    """One step in a decomposed action plan.

    ``action_text`` is fed verbatim to :meth:`TextToMotionSSM.stream_begin`
    as the text condition. ``until`` is the parsed termination predicate
    the runner evaluates each frame.
    """

    action_text: str
    until: ParsedCondition


class ActionPlanner:
    """Wrapper around the fine-tuned LM that emits structured action plans.

    Lazy-imports transformers so the rest of the agent module is usable
    in offline tests that don't need the trained checkpoint. Initialising
    the planner without a valid checkpoint path raises immediately.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        device: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.3,
    ) -> None:
        path = Path(checkpoint_dir)

        if not path.exists():
            raise FileNotFoundError(
                f"planner checkpoint not found at {path}. "
                "Train it with scripts/training/train_planner_lm.py first."
            )
        self.device = torch.device(
            "cuda" if (device == "auto" and torch.cuda.is_available()) or device == "cuda"
            else "cpu",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path).to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        log.info("[planner] loaded %s on %s", path, self.device)

    def __call__(self, instruction: str) -> list[PlannedAction]:
        """Decompose one instruction into a validated action plan.

        Raises ValueError if the LM emits something we can't parse -- the
        caller is responsible for falling back (retry, surface to the user,
        etc.). We do NOT silently drop bad actions; a malformed plan must
        bubble up so the demo doesn't run garbage past the renderer.
        """
        prompt = PROMPT_TEMPLATE.format(instruction=instruction)
        completion = self.generate_completion(prompt)

        return self.parse_completion(completion)

    def generate_completion(self, prompt: str) -> str:
        """Run the LM on the prompt and return only the completion portion."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # Strip the prompt from the generated output
        gen_ids = out[0, prompt_len:]
        completion = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        return completion

    def parse_completion(self, completion: str) -> list[PlannedAction]:
        """Parse the JSON action list out of the LM's completion + validate
        every ``until`` field against the runtime grammar.
        """
        # The LM may produce trailing text after the JSON; clip at the
        # first balanced closing bracket. JSON list opens with '[' and the
        # planner is trained to emit a list, so this is a tight cut.
        completion = completion.strip()

        if not completion.startswith("["):
            raise ValueError(
                f"planner output did not start with a JSON list: {completion[:80]!r}"
            )

        try:
            actions_raw = json.loads(completion)
        except json.JSONDecodeError:
            # Fallback: clip at first ']' and retry; the LM sometimes
            # generates after the close bracket.
            end_idx = completion.find("]")

            if end_idx > 0:
                try:
                    actions_raw = json.loads(completion[: end_idx + 1])
                except json.JSONDecodeError as e:
                    raise ValueError(f"planner emitted invalid JSON: {e}") from e
            else:
                raise ValueError(
                    f"no closing bracket found in planner output: {completion[:120]!r}"
                )

        if not isinstance(actions_raw, list):
            raise ValueError(
                f"planner emitted a non-list JSON value: {type(actions_raw).__name__}"
            )
        out: list[PlannedAction] = []

        for i, act in enumerate(actions_raw):
            if not isinstance(act, dict):
                raise ValueError(f"action {i} is not a dict: {act!r}")

            if "action" not in act or "until" not in act:
                raise ValueError(
                    f"action {i} missing required keys (action, until): {act!r}"
                )
            try:
                until_parsed = parse_condition(act["until"])
            except ValueError as e:
                raise ValueError(
                    f"action {i} has unparseable until={act['until']!r}: {e}"
                ) from e
            out.append(PlannedAction(action_text=str(act["action"]), until=until_parsed))

        return out
