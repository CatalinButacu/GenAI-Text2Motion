from __future__ import annotations

from text2motion.generation.contracts import StreamAnchors


def check_budget_values(steps: int, uniforms, anchor_every: int, anchor_context: int) -> None:
    if steps < 0:
        raise ValueError(f"steps must be >= 0 (0 means unbounded), got {steps}")
    if steps == 0 and uniforms is not None:
        raise ValueError("unbounded streaming cannot consume a fixed uniforms tensor")
    if anchor_every < 0 or anchor_context < 0:
        raise ValueError(
            f"anchor_every ({anchor_every}) and anchor_context ({anchor_context}) must be >= 0"
        )


def check_anchor_reach(
    name: str, limit: int, prefix_len: int, anchor_every: int, anchor_context: int
) -> None:
    available = limit - prefix_len
    deepest = anchor_context + anchor_every
    if deepest > available:
        raise ValueError(
            f"re-anchoring every {anchor_every} tokens over a {anchor_context}-token "
            f"context reaches position {prefix_len + deepest}, but "
            f"{name} has {limit} learned positions"
        )


def check_step_reach(name: str, limit: int, prefix_len: int, steps: int) -> None:
    available = limit - prefix_len
    if steps == 0:
        raise RuntimeError(
            f"{name} cannot stream unboundedly: it has "
            f"{limit} learned positions, so generation stops after {available} steps. "
            "Unbounded streaming requires a recurrent backbone (mamba); "
            "for this backbone pass a positive step budget instead."
        )
    if steps > available:
        raise ValueError(
            f"{name} supports at most {available} steps after a "
            f"{prefix_len}-token text prefix ({limit} learned positions), got {steps}"
        )


def check_anchor_request(anchors: StreamAnchors) -> None:
    if (anchors.text_at is not None or anchors.context_at is not None) and not anchors.every:
        raise ValueError(
            "text_at and anchor_context_at require re-anchoring; set anchor_every above 0"
        )
