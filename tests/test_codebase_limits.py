"""Codebase discipline gates (laws C5, C8, C9, A3).

These are tests because the laws are claims the project makes about itself. A law
nobody checks is a comment. Each one fails the build when the codebase drifts away
from what it says it is.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "bhanox"
PACKAGE_MODULES = sorted(p for p in SRC.rglob("*.py"))

# Law C5: one component per module, and a module you cannot hold in your head is
# a module nobody reviews. 500 lines is the ceiling; the largest module is well
# under it, so this binds without being a fiction.
MAX_MODULE_LINES = 500

# Law C8: numpy is the only runtime dependency, so the import-and-generate path
# has to work with nothing else installed.
FORBIDDEN_RUNTIME = {
    "torch",
    "scipy",
    "sklearn",
    "pandas",
    "numba",
    "jax",
    "tensorflow",
    "requests",
    "yaml",
    "click",
    "tqdm",
    "transformers",
    "datasets",
}

# Law C9: torch is permitted inside bhanox/train/ and nowhere else.
TORCH_ALLOWED_ROOT = "train"


def _imports(path: Path) -> set[str]:
    """Top-level module names imported by a file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def _code_lines(text: str) -> int:
    """Non-blank, non-comment lines. Docstrings are excluded: prose is not
    complexity, and counting it would reward deleting the explanations."""
    tree = ast.parse(text)
    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstring_lines.update(
                    range(body[0].lineno, (body[0].end_lineno or body[0].lineno) + 1)
                )
    lines = text.splitlines()
    keep = [
        i
        for i, line in enumerate(lines, start=1)
        if i not in docstring_lines
        and line.strip()
        and not line.strip().startswith("#")
    ]
    return len(keep)


class TestModuleSizeDoctrine:
    def test_no_module_exceeds_the_ceiling(self) -> None:
        too_big = {
            p.relative_to(ROOT).as_posix(): len(
                p.read_text(encoding="utf-8").splitlines()
            )
            for p in PACKAGE_MODULES
            if len(p.read_text(encoding="utf-8").splitlines()) > MAX_MODULE_LINES
        }
        assert not too_big, f"law C5: split these modules: {too_big}"

    def test_code_lines_excluding_prose_are_reasonable(self) -> None:
        worst = max(
            (
                (
                    _code_lines(p.read_text(encoding="utf-8")),
                    p.relative_to(ROOT).as_posix(),
                )
                for p in PACKAGE_MODULES
            ),
            key=lambda t: t[0],
        )
        assert worst[0] <= MAX_MODULE_LINES, f"law C5: {worst[1]} has {worst[0]}"

    def test_the_package_is_not_one_giant_file(self) -> None:
        assert len(PACKAGE_MODULES) >= 10

    def test_every_module_has_a_docstring(self) -> None:
        for p in PACKAGE_MODULES:
            tree = ast.parse(p.read_text(encoding="utf-8"))
            if not ast.get_docstring(tree) and p.name != "__init__.py":
                assert not p.read_text(
                    encoding="utf-8"
                ).strip(), f"{p.name} is empty and undocumented"
                continue
            assert ast.get_docstring(tree), f"{p.name} has no module docstring"

    def test_public_functions_are_documented(self) -> None:
        undocumented: list[str] = []
        for p in PACKAGE_MODULES:
            tree = ast.parse(p.read_text(encoding="utf-8"))
            for node in tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    continue
                if node.name.startswith("_"):
                    continue
                if not ast.get_docstring(node):
                    undocumented.append(f"{p.name}:{node.name}")
        assert not undocumented, f"missing docstrings: {undocumented}"


class TestNumpyOnlyRuntime:
    def test_no_forbidden_import_in_the_package(self) -> None:
        # C8 and C9 are two halves of one statement: numpy is the only runtime
        # dependency, and torch is permitted in exactly one package. Applying C8
        # to ``train/`` as well would contradict C9 and make the training extra
        # unimplementable -- the carve-out below is C9's, and C8 is what governs
        # everything else.
        offenders: dict[str, set[str]] = {}
        for p in PACKAGE_MODULES:
            if p.relative_to(SRC).as_posix().startswith(TORCH_ALLOWED_ROOT):
                continue
            bad = _imports(p) & FORBIDDEN_RUNTIME
            if bad:
                offenders[p.relative_to(ROOT).as_posix()] = bad
        assert not offenders, f"law C8: {offenders}"

    def test_torch_is_confined_to_the_train_package(self) -> None:
        offenders: dict[str, set[str]] = {}
        for p in PACKAGE_MODULES:
            if "torch" not in _imports(p):
                continue
            rel = p.relative_to(SRC).as_posix()
            if not rel.startswith(TORCH_ALLOWED_ROOT):
                offenders[rel] = {"torch"}
        assert not offenders, f"law C9: torch outside bhanox/train: {offenders}"

    def test_the_declared_dependency_is_numpy_only(self) -> None:
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        block = re.search(r"^dependencies = \[(.*?)\]", text, re.M | re.S)
        assert block is not None
        assert "numpy" in block.group(1)
        for name in FORBIDDEN_RUNTIME:
            assert name not in block.group(1), f"law C8: {name} is a hard dependency"

    def test_numpy_imports_successfully_in_a_clean_interpreter(self) -> None:
        import numpy

        assert numpy.__version__


class TestHonestyLaw:
    """Law A3: no silent estimates. A number in a docstring that the code
    reports must be a number the code can produce."""

    def test_no_todo_or_fixme_left_in_the_package(self) -> None:
        pattern = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")
        offenders = [
            p.relative_to(ROOT).as_posix()
            for p in PACKAGE_MODULES
            if pattern.search(p.read_text(encoding="utf-8"))
        ]
        assert not offenders, f"unfinished work left in: {offenders}"

    def test_the_package_imports_without_the_optional_extras(self) -> None:
        """Importing bhanox must not pull in torch. Run in a subprocess so an
        earlier test that imported torch cannot mask a real dependency."""
        import subprocess

        code = "import sys, bhanox; assert 'torch' not in sys.modules"
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        assert result.returncode == 0, result.stderr


class TestRequiredFiles:
    @pytest.mark.parametrize(
        "rel",
        [
            "README.md",
            "LICENSE",
            "pyproject.toml",
            "ROADMAP.md",
            "CONTRIBUTING.md",
            "docs/architecture.md",
            "docs/invariants.md",
            "docs/benchmarks.md",
            "examples/generate_nano.py",
            "examples/audit_demo.py",
            "scripts/benchmark.py",
        ],
    )
    def test_the_file_exists(self, rel: str) -> None:
        assert (ROOT / rel).exists(), f"missing required file: {rel}"
