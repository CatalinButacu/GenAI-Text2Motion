from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from text2motion.common.fingerprint import recipe_hash

type Run = Callable[[dict[str, Path], Path], None]


@dataclass(frozen=True)
class Step:
    name: str
    inputs: tuple[str, ...]
    output: str
    run: Run


def run_cached_steps(
    steps: Sequence[Step], root: Path, context: Mapping[str, object]
) -> dict[str, Path]:
    recipe = context["recipe"]
    if not isinstance(recipe, Mapping):
        raise TypeError("context['recipe'] must be a mapping")
    stamp = recipe_hash(recipe)
    names = {step.name for step in steps}
    for step in steps:
        for needed in step.inputs:
            if needed not in names:
                raise KeyError(f"step {step.name} needs {needed}, which no step produces")
    root.mkdir(parents=True, exist_ok=True)
    done: dict[str, Path] = {}
    remaining = list(steps)
    while remaining:
        ready = [step for step in remaining if all(name in done for name in step.inputs)]
        if not ready:
            raise ValueError("steps form a cycle")
        step = ready[0]
        remaining.remove(step)
        output = root / step.output
        marker = root / f"{step.output}.{stamp}"
        if not marker.exists():
            step.run({name: done[name] for name in step.inputs}, output)
            marker.write_text(stamp, encoding="utf-8")
        done[step.name] = output
    return done
