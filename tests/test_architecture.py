import ast
import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("path", sorted(Path("src/taco_shell_rag/core").glob("*.py")))
def test_core_depends_only_on_standard_library_and_core_protocols(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            name = "." * node.level + (node.module or "")
            modules = [importlib.util.resolve_name(name, "taco_shell_rag.core")]
        else:
            continue
        for module in modules:
            assert module.split(".")[0] in sys.stdlib_module_names or (
                module == "taco_shell_rag.core" or module.startswith("taco_shell_rag.core.")
            ), f"{path}: core must not import {module}"
