#!/usr/bin/env python
"""Parse-output cache for eval_prompt_following.

Caches the result of pipeline.parse_and_plan() keyed by (prompt, checkpoint, device).
Allows re-scoring with a changed scorer without re-running the parser.

Cache layout
------------
  data/.cache/pf_parse/<sha256>.json

Each file stores:
  {
    "prompt": "...",
    "checkpoint": "...",
    "device": "...",
    "parsed_actions": [...],
    "parsed_entities": [...]
  }
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

log = logging.getLogger(__name__)

CACHE_DIR = Path("data/.cache/pf_parse")


def cache_key(prompt: str, checkpoint_path: str, device: str) -> str:
    payload = f"{prompt}|{checkpoint_path}|{device}"
    return hashlib.sha256(payload.encode()).hexdigest()


def load_from_cache(key: str, cache_dir: Path) -> SimpleNamespace | None:
    cache_path = cache_dir / f"{key}.json"

    if not cache_path.exists():
        return None

    try:
        stored = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Cache read failed for %s: %s", cache_path, exc)
        return None

    actions = [SimpleNamespace(verb=a) for a in stored.get("parsed_actions", [])]
    entities = [SimpleNamespace(label=e) for e in stored.get("parsed_entities", [])]
    return SimpleNamespace(actions=actions, entities=entities)


def save_to_cache(
    key: str,
    prompt: str,
    checkpoint_path: str,
    device: str,
    parsedScene: object,
    cache_dir: Path,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{key}.json"

    raw_actions = getattr(parsedScene, "actions", []) or []
    raw_entities = getattr(parsedScene, "entities", []) or []

    parsed_actions = []

    for act in raw_actions:
        verb = getattr(act, "verb", None) or getattr(act, "action", None) or str(act)
        parsed_actions.append(str(verb).lower().strip())

    parsed_entities = []

    for ent in raw_entities:
        label = getattr(ent, "label", None) or getattr(ent, "name", None) or str(ent)
        parsed_entities.append(str(label).lower().strip())

    entry = {
        "prompt": prompt,
        "checkpoint": checkpoint_path,
        "device": device,
        "parsed_actions": parsed_actions,
        "parsed_entities": parsed_entities,
    }

    try:
        cache_path.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        log.warning("Cache write failed for %s: %s", cache_path, exc)


def load_or_run(
    prompt: str,
    checkpoint_path: str,
    device: str,
    run_fn,
    use_cache: bool = True,
    cache_dir: Path = CACHE_DIR,
) -> object:
    """Return a parsed-scene namespace, using cache when available.

    Parameters
    ----------
    prompt:
        The text prompt to parse.
    checkpoint_path:
        Checkpoint used by the pipeline (part of cache key).
    device:
        Device string (part of cache key).
    run_fn:
        Callable(prompt) -> parsed_scene.  Called on cache miss.
    use_cache:
        Set False to bypass cache (equivalent to --no-cache).
    cache_dir:
        Directory for cache files.
    """
    if not use_cache:
        return run_fn(prompt)

    key = cache_key(prompt, checkpoint_path, device)
    cached = load_from_cache(key, cache_dir)

    if cached is not None:
        log.debug("cache hit for prompt %.40r", prompt)
        return cached

    result = run_fn(prompt)

    if result is not None:
        save_to_cache(key, prompt, checkpoint_path, device, result, cache_dir)

    return result
