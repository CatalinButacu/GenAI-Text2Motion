# python scripts/maintenance/add_class_separators.py
import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "src" / "text2motion"
RULE = "#" * 60
LABEL_PREFIX = "#" + " " * 7
SEPARATOR_LINE_COUNT = 5


def separator(class_name: str) -> list[str]:
    return [RULE, f"{LABEL_PREFIX}{class_name}", RULE, "", ""]


def class_start_lines(tree: ast.Module) -> list[tuple[int, str]]:
    starts = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        decorator_lines = [decorator.lineno for decorator in node.decorator_list]
        starts.append((min([node.lineno, *decorator_lines]), node.name))
    return starts


def has_separator(lines: list[str], index: int) -> bool:
    if index < SEPARATOR_LINE_COUNT:
        return False
    above = lines[index - SEPARATOR_LINE_COUNT : index]
    return above[0] == RULE and above[2] == RULE


def separated_text(text: str) -> str:
    starts = class_start_lines(ast.parse(text))
    if not starts:
        return text
    ending = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(ending)
    for first_line, class_name in reversed(starts):
        index = first_line - 1
        if has_separator(lines, index):
            lines[index - SEPARATOR_LINE_COUNT : index] = separator(class_name)
            continue
        lines[index:index] = separator(class_name)
    return ending.join(lines)


def separate_classes(root: Path) -> list[str]:
    changed = []
    for path in sorted(root.rglob("*.py")):
        original = path.read_bytes().decode("utf-8")
        updated = separated_text(original)
        if updated == original:
            continue
        path.write_bytes(updated.encode("utf-8"))
        changed.append(path.relative_to(root).as_posix())
    return changed


def main() -> int:
    changed = separate_classes(SOURCE)
    print(f"class separators: {len(changed)} files updated")
    for name in changed:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
