import json
from pathlib import Path

from text2motion.app.cli import COMMANDS, register_command_enums

GUARD_FILE = Path(__file__).with_name("cli_schema_guard.json")


def measure() -> dict:
    register_command_enums()
    schemas = {
        name: command.request_model.model_json_schema() for name, command in COMMANDS.items()
    }
    return json.loads(json.dumps(schemas).replace(r"\\", "/"))


def test_every_cli_request_matches_its_recorded_json_schema():
    recorded = json.loads(GUARD_FILE.read_text(encoding="utf-8"))
    measured = measure()

    assert list(measured) == list(recorded)
    for name, schema in recorded.items():
        assert json.dumps(measured[name]) == json.dumps(schema), f"{name}: schema changed"
