"""Check local module paths without importing the simulator dependencies.

Run standalone with: python -m unittest discover -s tests -p test_local_imports.py
"""

import ast
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


class LocalImportTests(unittest.TestCase):
    def test_all_local_import_modules_exist(self):
        for folder in (SRC, ROOT / "tests", ROOT / "scripts"):
            for path in folder.rglob("*.py"):
                tree = ast.parse(path.read_text(), filename=str(path))
                package = ""
                if path.is_relative_to(SRC):
                    package = ".".join(path.relative_to(SRC).parts[:-1])
                for node in ast.walk(tree):
                    modules = []
                    if isinstance(node, ast.Import):
                        modules = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        if node.level:
                            if not package:
                                continue
                            module = importlib.util.resolve_name("." * node.level + module, package)
                        modules = [module]
                    for module in modules:
                        if not module.startswith("mjlab_microduck"):
                            continue
                        target = SRC.joinpath(*module.split("."))
                        with self.subTest(file=str(path.relative_to(ROOT)), line=node.lineno, module=module):
                            self.assertTrue(
                                target.with_suffix(".py").is_file() or (target / "__init__.py").is_file(),
                                f"Missing local module: {module}",
                            )

    def test_generic_package_has_no_concrete_task_imports(self):
        generic = SRC / "mjlab_microduck/tasks/locomotion"
        prefix = "mjlab_microduck.tasks."
        for path in generic.rglob("*.py"):
            package = ".".join(path.relative_to(SRC).parts[:-1])
            for node in ast.walk(ast.parse(path.read_text())):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if node.level:
                        module = importlib.util.resolve_name("." * node.level + module, package)
                    modules = [module]
                for module in modules:
                    if module.startswith(prefix):
                        self.assertTrue(module.startswith(prefix + "locomotion"), module)


if __name__ == "__main__":
    unittest.main()
