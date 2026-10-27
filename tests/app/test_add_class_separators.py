import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "maintenance" / "add_class_separators.py"
SPEC = importlib.util.spec_from_file_location("add_class_separators", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
add_class_separators = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(add_class_separators)

RULE = "#" * 60
TWO_CLASSES = "import os\n\n\nclass Alpha:\n    pass\n\n\n@decorate\nclass Beta:\n    pass\n"
SEPARATED = (
    f"import os\n\n\n{RULE}\n#       Alpha\n{RULE}\n\n\nclass Alpha:\n    pass\n\n\n"
    f"{RULE}\n#       Beta\n{RULE}\n\n\n@decorate\nclass Beta:\n    pass\n"
)


def test_each_class_gets_a_separator_above_its_first_decorator():
    assert add_class_separators.separated_text(TWO_CLASSES) == SEPARATED


def test_a_second_run_changes_nothing():
    assert add_class_separators.separated_text(SEPARATED) == SEPARATED


def test_a_stale_label_takes_the_current_class_name():
    stale = SEPARATED.replace("#       Alpha", "#       OldAlpha")
    assert add_class_separators.separated_text(stale) == SEPARATED


def test_a_file_with_one_class_gets_a_separator():
    single = "class Alpha:\n    pass\n"
    expected = f"{RULE}\n#       Alpha\n{RULE}\n\n\nclass Alpha:\n    pass\n"
    assert add_class_separators.separated_text(single) == expected


def test_a_file_without_classes_stays_unchanged():
    functions = "def alpha():\n    pass\n"
    assert add_class_separators.separated_text(functions) == functions


def test_windows_line_endings_stay():
    windows = TWO_CLASSES.replace("\n", "\r\n")
    assert add_class_separators.separated_text(windows) == SEPARATED.replace("\n", "\r\n")
