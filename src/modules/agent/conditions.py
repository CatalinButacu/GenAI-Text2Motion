"""Termination-predicate parser for the agent runner.

The planner LM emits action plans where each action carries an ``until``
string in a small grammar this module understands:

  - ``"duration(N)"``         -- elapsed-frame threshold (N frames; 20 fps default)
  - ``"distance(obj) < D"``   -- avatar-to-scene-object distance below D meters
  - ``"distance(obj) > D"``   -- distance above D meters (e.g., walk away from)
  - ``"rotated(D)"``           -- avatar yaw has changed by D degrees since action start
  - ``"completed"``            -- always fires once, used for atomic actions like waves

Parsing happens once when the action is dequeued; the parsed result is a
callable taking a :class:`WorldState` and returning bool. The grammar is
intentionally restricted so the GPT-2-small planner has fewer ways to
emit invalid output; the runner rejects an action whose ``until`` cannot
be parsed.

The grammar is also what the ``outlines`` library will enforce at
generation time so the LM never emits unparseable strings in the first
place; see ``scripts/training/train_planner_lm.py`` (TBD).
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from .world_state import WorldState

Predicate = Callable[[WorldState], bool]


@dataclass
class ParsedCondition:
    """Parsed termination predicate + a human-readable echo of the source.

    The echo is kept so the HUD overlay in the demo can show exactly what
    condition the runner is currently waiting on, character-for-character
    matching what the planner emitted.
    """

    source: str
    callable_: Predicate

    def __call__(self, world: WorldState) -> bool:
        return self.callable_(world)


# Compiled regexes for each grammar variant. Order matters when multiple
# patterns could match (e.g. distance with < vs >); we test most specific
# patterns first.
_DURATION_RE = re.compile(r"^duration\(\s*(\d+(?:\.\d+)?)\s*\)$")
_DIST_LT_RE = re.compile(r"^distance\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*<\s*(\d+(?:\.\d+)?)$")
_DIST_GT_RE = re.compile(r"^distance\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*>\s*(\d+(?:\.\d+)?)$")
_ROTATED_RE = re.compile(r"^rotated\(\s*(\d+(?:\.\d+)?)\s*\)$")
_COMPLETED_RE = re.compile(r"^completed$")


def parse_condition(source: str) -> ParsedCondition:
    """Parse a single termination-predicate string.

    Raises ``ValueError`` for any string outside the grammar so the runner
    can reject the whole action plan up-front rather than failing mid-clip.
    """
    s = source.strip()

    if (m := _COMPLETED_RE.match(s)):
        return ParsedCondition(source=s, callable_=_make_completed())

    if (m := _DURATION_RE.match(s)):
        n = float(m.group(1))
        return ParsedCondition(source=s, callable_=_make_duration(n))

    if (m := _DIST_LT_RE.match(s)):
        obj, threshold = m.group(1), float(m.group(2))
        return ParsedCondition(source=s, callable_=_make_distance_lt(obj, threshold))

    if (m := _DIST_GT_RE.match(s)):
        obj, threshold = m.group(1), float(m.group(2))
        return ParsedCondition(source=s, callable_=_make_distance_gt(obj, threshold))

    if (m := _ROTATED_RE.match(s)):
        deg = float(m.group(1))
        return ParsedCondition(source=s, callable_=_make_rotated(deg))
    raise ValueError(
        f"unrecognised termination condition: {source!r}. "
        "Grammar: duration(N) | distance(obj) (< | >) D | rotated(D) | completed"
    )


# ---------------------------------------------------------------------- #
# Predicate factories -- each returns a Callable[[WorldState], bool].     #
# Kept as small closures so the parse-time arguments are baked in and the #
# runtime call signature is uniform.                                       #
# ---------------------------------------------------------------------- #


def _make_completed() -> Predicate:
    """One-shot fire: always returns True. The runner expects this when the
    planner emits an atomic action (wave, kick, etc.) -- the action runs
    once and the runner advances.
    """

    def _p(world: WorldState) -> bool:
        return True

    return _p


def _make_duration(n: float) -> Predicate:
    threshold = int(round(n))

    def _p(world: WorldState) -> bool:
        return world.frames_in_action >= threshold

    return _p


def _make_distance_lt(obj: str, threshold: float) -> Predicate:
    def _p(world: WorldState) -> bool:
        return world.distance_to(obj) < threshold

    return _p


def _make_distance_gt(obj: str, threshold: float) -> Predicate:
    def _p(world: WorldState) -> bool:
        return world.distance_to(obj) > threshold

    return _p


def _make_rotated(deg: float) -> Predicate:
    def _p(world: WorldState) -> bool:
        return world.rotated_since_start_deg() >= deg

    return _p
