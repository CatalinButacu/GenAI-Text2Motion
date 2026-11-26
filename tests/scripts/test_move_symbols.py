import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "maintenance" / "move_symbols.py"


def mover(monkeypatch):
    spec = importlib.util.spec_from_file_location("move_symbols", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_repo(root: Path) -> None:
    package = root / "src" / "text2motion"
    write(package / "__init__.py", "")
    write(package / "app" / "__init__.py", "")
    write(
        package / "app" / "service.py",
        "import json\nfrom text2motion.base import Base\n\nLIMIT = 3\n\n\ndef encode(value):\n    return json.dumps(value)[:LIMIT]\n\n\n"
        "# box\n@staticmethod\ndef helper():\n    return 1\n\n\nclass Server:\n    def run(self):\n        return encode(2)\n",
    )
    write(
        root / "tests" / "test_service.py",
        "from text2motion.app.service import Base, Server, encode as enc\n\n"
        "TARGET = 'text2motion.app.service.encode'\nCOMMAND = 'text2motion.app.service:Server'\n",
    )
    write(root / "scripts" / "use.py", "from text2motion.app import service\n")


def test_a_split_moves_code_rewrites_imports_and_deletes_the_empty_file(monkeypatch, tmp_path):
    module = mover(monkeypatch)
    make_repo(tmp_path)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            [
                {
                    "from": "text2motion.app.service",
                    "to": "text2motion.app.service.codec",
                    "names": ["LIMIT", "encode", "helper"],
                },
                {
                    "from": "text2motion.app.service",
                    "to": "text2motion.app.service.server",
                    "names": ["Server"],
                },
            ]
        ),
        encoding="utf-8",
    )

    warnings = module.apply_moves(tmp_path, module.load_moves(plan))

    package = tmp_path / "src" / "text2motion" / "app" / "service"
    assert not (tmp_path / "src" / "text2motion" / "app" / "service.py").exists()
    assert (package / "__init__.py").exists()
    codec = (package / "codec.py").read_text(encoding="utf-8")
    assert "import json" in codec and "# box" in codec and "@staticmethod" in codec
    server = (package / "server.py").read_text(encoding="utf-8")
    assert "from text2motion.app.service.codec import encode" in server
    assert "import json" not in server
    test_text = (tmp_path / "tests" / "test_service.py").read_text(encoding="utf-8")
    assert "from text2motion.app.service.codec import encode as enc" in test_text
    assert "from text2motion.app.service.server import Server" in test_text
    assert "from text2motion.base import Base" in test_text
    assert '"text2motion.app.service.codec.encode"' in test_text
    assert '"text2motion.app.service.server:Server"' in test_text
    assert warnings == ["scripts/use.py:1 imports module text2motion.app.service"]


def test_an_unknown_name_fails_loud(monkeypatch, tmp_path):
    module = mover(monkeypatch)
    make_repo(tmp_path)
    move = module.Move("text2motion.app.service", "text2motion.app.codec", ("missing",))

    try:
        module.apply_moves(tmp_path, [move])
    except module.MoveError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("expected MoveError")


def test_a_target_that_imported_the_moved_names_does_not_import_itself(monkeypatch, tmp_path):
    module = mover(monkeypatch)
    package = tmp_path / "src" / "text2motion"
    write(package / "__init__.py", "")
    write(package / "a.py", "from text2motion.b import B\n\n\ndef make():\n    return B()\n")
    write(
        package / "b.py",
        "from text2motion.a import make\n\n\nclass B:\n    def again(self):\n        return make()\n",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "scripts").mkdir()

    module.apply_moves(tmp_path, [module.Move("text2motion.a", "text2motion.b", ("make",))])

    target = (package / "b.py").read_text(encoding="utf-8")
    assert "from text2motion.b import" not in target
    assert "def make" in target
    assert not (package / "a.py").exists()
