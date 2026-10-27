RULE = "#" * 60
TWO_CLASSES = "import os\n\n\nclass Alpha:\n    pass\n\n\n@decorate\nclass Beta:\n    pass\n"
SEPARATED = (
    f"import os\n\n\n{RULE}\n#       Alpha\n{RULE}\nclass Alpha:\n    pass\n\n\n"
    f"{RULE}\n#       Beta\n{RULE}\n@decorate\nclass Beta:\n    pass\n"
)


def test_each_class_gets_a_separator_above_its_first_decorator(add_class_separators):
    assert add_class_separators.separated_text(TWO_CLASSES) == SEPARATED


def test_a_second_run_changes_nothing(add_class_separators):
    assert add_class_separators.separated_text(SEPARATED) == SEPARATED


def test_a_stale_label_takes_the_current_class_name(add_class_separators):
    stale = SEPARATED.replace("#       Alpha", "#       OldAlpha")
    assert add_class_separators.separated_text(stale) == SEPARATED


def test_blank_lines_between_an_old_separator_and_its_class_go_away(add_class_separators):
    spaced = SEPARATED.replace(f"{RULE}\nclass Alpha", f"{RULE}\n\n\nclass Alpha")
    spaced = spaced.replace(f"{RULE}\n@decorate", f"{RULE}\n\n@decorate")
    assert add_class_separators.separated_text(spaced) == SEPARATED


def test_a_file_with_one_class_gets_a_separator(add_class_separators):
    single = "class Alpha:\n    pass\n"
    expected = f"{RULE}\n#       Alpha\n{RULE}\nclass Alpha:\n    pass\n"
    assert add_class_separators.separated_text(single) == expected


def test_a_file_without_classes_stays_unchanged(add_class_separators):
    functions = "def alpha():\n    pass\n"
    assert add_class_separators.separated_text(functions) == functions


def test_windows_line_endings_stay(add_class_separators):
    windows = TWO_CLASSES.replace("\n", "\r\n")
    assert add_class_separators.separated_text(windows) == SEPARATED.replace("\n", "\r\n")
