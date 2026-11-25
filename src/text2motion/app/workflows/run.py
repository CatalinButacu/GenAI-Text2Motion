from __future__ import annotations

import logging
from typing import Any

from text2motion.app.workflows.requests import RunRequest
from text2motion.app.workflows.steps import workflow_steps
from text2motion.common.fingerprint import recipe_hash
from text2motion.common.manifest import write_manifest
from text2motion.common.runner import run_steps

MANIFEST_NAME = "manifest.json"

logger = logging.getLogger(__name__)


def workflow_recipe(request: RunRequest) -> dict[str, object]:
    return request.model_dump(mode="json", exclude={"root"})


def run_workflow(app: Any, request: RunRequest) -> None:
    steps = workflow_steps(app, request)
    recipe = workflow_recipe(request)
    done = run_steps(steps, request.root, {"recipe": recipe})
    stamp = recipe_hash(recipe)
    entries = {name: {"recipe_hash": stamp, "output": path.name} for name, path in done.items()}
    write_manifest(request.root / MANIFEST_NAME, recipe, {"steps": entries})
    logger.info(f"workflow {request.workflow.value}: {sorted(entries)} recipe {stamp}")
