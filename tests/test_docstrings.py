"""Enforce complete NumPy documentation and function annotations."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path


ROOT = Path(__file__).parents[1]
PYTHON_ROOTS = (ROOT / 'paperminer', ROOT / 'tests')
DEFINITION_TYPES = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
NON_NUMPY_HEADINGS = re.compile(
    r'^\s*(Args|Arguments|Asserts|Attributes|Returns|Yields|Raises|Warns):\s*$',
    re.MULTILINE,
)


def _python_files() -> Iterator[Path]:
    """Yield every maintained Python file.

    Yields
    ------
    pathlib.Path
        Python source or test file.
    """
    for root in PYTHON_ROOTS:
        yield from sorted(root.rglob('*.py'))


def _definitions(tree: ast.Module) -> Iterator[tuple[ast.AST, str]]:
    """Yield every documentable definition in a syntax tree.

    Parameters
    ----------
    tree : ast.Module
        Parsed Python module.

    Yields
    ------
    tuple of ast.AST and str
        Definition node and its display name.
    """
    yield tree, '<module>'
    for node in ast.walk(tree):
        if isinstance(node, DEFINITION_TYPES):
            yield node, node.name


def test_every_python_definition_has_a_docstring() -> None:
    """Require docstrings on modules, classes, functions, and methods."""
    missing = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node, name in _definitions(tree):
            if ast.get_docstring(node, clean=False) is None:
                line = getattr(node, 'lineno', 1)
                missing.append(f'{path.relative_to(ROOT)}:{line}: {name}')

    assert not missing, 'Missing docstrings:\n' + '\n'.join(missing)


def test_docstrings_do_not_use_non_numpy_section_headings() -> None:
    """Reject Google-style and custom section headings."""
    invalid = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node, name in _definitions(tree):
            docstring = ast.get_docstring(node, clean=False) or ''
            if NON_NUMPY_HEADINGS.search(docstring):
                line = getattr(node, 'lineno', 1)
                invalid.append(f'{path.relative_to(ROOT)}:{line}: {name}')

    assert not invalid, 'Non-NumPy docstring headings:\n' + '\n'.join(invalid)


def test_every_python_function_has_complete_type_annotations() -> None:
    """Require parameter and return annotations on every function and method."""
    missing = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            relative_path = path.relative_to(ROOT)
            arguments = [
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            ]
            for argument in arguments:
                if argument.arg not in {'self', 'cls'} and argument.annotation is None:
                    missing.append(
                        f'{relative_path}:{node.lineno}: {node.name} parameter {argument.arg}'
                    )
            if node.args.vararg is not None and node.args.vararg.annotation is None:
                missing.append(
                    f'{relative_path}:{node.lineno}: {node.name} parameter *{node.args.vararg.arg}'
                )
            if node.args.kwarg is not None and node.args.kwarg.annotation is None:
                missing.append(
                    f'{relative_path}:{node.lineno}: {node.name} parameter **{node.args.kwarg.arg}'
                )
            if node.returns is None:
                missing.append(f'{relative_path}:{node.lineno}: {node.name} return')

    assert not missing, 'Missing type annotations:\n' + '\n'.join(missing)
