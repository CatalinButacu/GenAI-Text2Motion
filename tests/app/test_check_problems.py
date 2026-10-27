import ast


def __rule_codes(check_problems, source: str) -> list[str]:
    problems = check_problems.module_name_problems(ast.parse(source))
    return [message.split(" ")[0] for _, message in problems]


def test_a_mutable_module_constant_is_reported(check_problems):
    source = "FEET = [7, 8]\nNAMES = {'a': 1}\nKEYS = {'a'}\nCOUNTS = dict(a=1)\n"
    assert __rule_codes(check_problems, source) == ["G01", "G01", "G01", "G01"]


def test_a_read_only_module_constant_passes(check_problems):
    source = (
        "from types import MappingProxyType\n"
        "FEET = (7, 8)\n"
        "KEYS = frozenset({'a'})\n"
        "NAMES = MappingProxyType({'a': 1})\n"
        "_FPS = 20\n"
        "type Pair = tuple[float, float]\n"
    )
    assert __rule_codes(check_problems, source) == []


def test_a_lowercase_module_name_is_reported(check_problems):
    assert __rule_codes(check_problems, "loaded_module = object()\n") == ["G02"]


def test_protocol_names_pass(check_problems):
    source = (
        "import logging\n"
        "logger = logging.getLogger(__name__)\n"
        "pytestmark = None\n"
        "__all__ = ['logger']\n"
    )
    assert __rule_codes(check_problems, source) == []


def test_names_inside_functions_are_not_module_state(check_problems):
    source = "def build():\n    names = []\n    return names\n"
    assert __rule_codes(check_problems, source) == []
