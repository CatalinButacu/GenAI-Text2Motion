import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "maintenance" / "rename_everywhere.py"
SPEC = importlib.util.spec_from_file_location("rename_everywhere", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
rename_everywhere = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rename_everywhere)


def test_a_file_rename_reaches_code_documents_and_the_assistant_folder(tmp_path):
    code = tmp_path / "src" / "text2motion" / "generation" / "trainer.py"
    note = tmp_path / "doc" / "note.md"
    agent = tmp_path / ".claude" / "agents" / "architect.md"
    history = tmp_path / ".claude" / "docs" / "STATUS.md"
    for path in (code, note, agent, history):
        path.parent.mkdir(parents=True)
    (code.parent / "fresh.py").write_text("", encoding="utf-8")
    code.write_text("from text2motion.generation.stale import X\n", encoding="utf-8")
    note.write_text("see `src/text2motion/generation/stale.py:12`\n", encoding="utf-8")
    agent.write_text("owns generation/stale.py and motion/stale.py\n", encoding="utf-8")
    history.write_text("renamed generation/stale.py\n", encoding="utf-8")
    old = "src/text2motion/generation/stale.py"
    new = "src/text2motion/generation/fresh.py"
    changed, kept = rename_everywhere.rename_in_text(tmp_path, old, new)
    assert sorted(changed) == [
        ".claude/agents/architect.md",
        "doc/note.md",
        "src/text2motion/generation/trainer.py",
    ]
    assert kept == [".claude/docs/STATUS.md"]
    assert code.read_text(encoding="utf-8") == "from text2motion.generation.fresh import X\n"
    assert "generation/fresh.py:12" in note.read_text(encoding="utf-8")
    assert agent.read_text(encoding="utf-8") == "owns generation/fresh.py and motion/stale.py\n"
    assert rename_everywhere.missing_modules(tmp_path) == []


def test_a_symbol_rename_matches_whole_words_only(tmp_path):
    code = tmp_path / "a.py"
    code.write_text("load_it()\nreload_it()\n", encoding="utf-8")
    rename_everywhere.rename_in_text(tmp_path, "load_it", "read_it")
    assert code.read_text(encoding="utf-8") == "read_it()\nreload_it()\n"


def test_a_limited_rename_changes_only_the_named_files(tmp_path):
    named = tmp_path / "quantizers.py"
    other = tmp_path / "contracts.py"
    named.write_text("class FSQ:\n    pass\n", encoding="utf-8")
    other.write_text("FSQ = 'fsq'\n", encoding="utf-8")
    changed, _ = rename_everywhere.rename_in_text(
        tmp_path, "FSQ", "FiniteScalarQuantizer", ["quantizers.py"]
    )
    assert changed == ["quantizers.py"]
    assert named.read_text(encoding="utf-8") == "class FiniteScalarQuantizer:\n    pass\n"
    assert other.read_text(encoding="utf-8") == "FSQ = 'fsq'\n"


def test_a_limited_rename_refuses_a_missing_file(tmp_path):
    try:
        rename_everywhere.rename_in_text(tmp_path, "FSQ", "FiniteScalarQuantizer", ["absent.py"])
    except FileNotFoundError as error:
        assert "absent.py" in str(error)
        return
    raise AssertionError("expected FileNotFoundError for a missing --only file")


def test_an_import_of_a_removed_module_is_reported(tmp_path):
    code = tmp_path / "src" / "text2motion" / "app" / "cli.py"
    code.parent.mkdir(parents=True)
    code.write_text("from text2motion.app.removed import load_it\n", encoding="utf-8")
    assert rename_everywhere.missing_modules(tmp_path) == [
        "src/text2motion/app/cli.py: text2motion.app.removed"
    ]
