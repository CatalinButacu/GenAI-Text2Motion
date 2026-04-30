from __future__ import annotations

import re

SKIP_ACTIONS = {"male", "female", "female1", "male1", "subj calibration"}


def normalizeParent(parts: list[str]) -> str:
    """Return a cleaned subject-context string from the parent directory, or ''."""
    if len(parts) < 2:
        return ""

    parent = re.sub(r"_c3d$", "", parts[-2], flags=re.IGNORECASE)
    parent = re.sub(r"^(Female|Male)\d*", "", parent)
    parent = re.sub(
        r"\s+", " ",
        re.sub(r"(?<=[a-z])(?=[A-Z])", " ", parent).replace("_", " "),
    ).strip().lower()

    return parent


def textFromFilename(relPath: str) -> str:
    parts = relPath.replace("\\", "/").split("/")
    fname = parts[-1]
    fname = re.sub(r"_stage[iv]+$", "", fname, flags=re.IGNORECASE)
    fname = re.sub(r"_c3d$", "", fname, flags=re.IGNORECASE)
    m = re.match(r"^[A-Z]?\d+\s*[-_]+\s*(.+)$", fname)

    if m:
        fname = m.group(1)
    fname = re.sub(r"^\d{3,}_", "", fname)
    fname = re.sub(r"_?\d{1,3}$", "", fname)
    action = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", fname)
    action = re.sub(r"\s+", " ", action.replace("_", " ").replace("-", " ")).strip().lower()

    if not action or re.match(r"^[\d\s]+$", action) or action in SKIP_ACTIONS:
        parent = normalizeParent(parts)

        return f"person {parent}" if parent and not re.match(r"^[\d\s]*$", parent) else ""

    parent = normalizeParent(parts)
    context = (
        f"person {parent}: "
        if parent and parent != action and not re.match(r"^(s?\d+|\d+)$", parent)
        else ""
    )

    return f"{context}{action}" if context else action

