#!/usr/bin/env python3
"""Fallback checker for Python.

Rules:
- UFB001: provably unnecessary fallback (`<truthy_literal> or <fallback>`)
- SFB001: suspicious fallback (`except` branch returns success)
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

UNNECESSARY_RULE_ID = "UFB001"
UNNECESSARY_RULE_MSG = "provably unnecessary fallback: left side of `or` is always truthy"
SUSPICIOUS_RULE_ID = "SFB001"
SUSPICIOUS_RULE_MSG = "suspicious fallback: `except` branch returns success"
IGNORE_UNNECESSARY_TOKEN = "fallbacklint: ignore[provably-unnecessary-fallback]"
IGNORE_SUSPICIOUS_TOKEN = "fallbacklint: ignore[suspicious-fallback]"


def is_definitely_truthy(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return bool(node.value)

    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return len(node.elts) > 0

    if isinstance(node, ast.Dict):
        return len(node.keys) > 0

    return False


def is_ignored(lines: list[str], lineno: int, token: str) -> bool:
    if lineno < 1 or lineno > len(lines):
        return False

    return token in lines[lineno - 1]


def iter_stmt_nodes(statements: list[ast.stmt]) -> list[ast.AST]:
    stack: list[ast.AST] = list(statements)
    result: list[ast.AST] = []

    while stack:
        current = stack.pop()
        result.append(current)

        for child in ast.iter_child_nodes(current):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            stack.append(child)

    return result


def find_returns(statements: list[ast.stmt]) -> list[ast.Return]:
    returns: list[ast.Return] = []
    for node in iter_stmt_nodes(statements):
        if isinstance(node, ast.Return):
            returns.append(node)
    return returns


def has_raise(statements: list[ast.stmt]) -> bool:
    return any(isinstance(node, ast.Raise) for node in iter_stmt_nodes(statements))


class FallbackVisitor(ast.NodeVisitor):
    def __init__(self, file_path: Path, lines: list[str]) -> None:
        self.file_path = file_path
        self.lines = lines
        self.issues: list[tuple[int, int, str]] = []

    def visit_BoolOp(self, node: ast.BoolOp) -> None:  # noqa: N802
        if isinstance(node.op, ast.Or) and len(node.values) >= 2:
            left = node.values[0]
            if is_definitely_truthy(left) and not is_ignored(self.lines, node.lineno, IGNORE_UNNECESSARY_TOKEN):
                self.issues.append((node.lineno, node.col_offset + 1, f"{UNNECESSARY_RULE_ID} {UNNECESSARY_RULE_MSG}"))

        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:  # noqa: N802
        if find_returns(node.body):
            for handler in node.handlers:
                if has_raise(handler.body):
                    continue

                for return_stmt in find_returns(handler.body):
                    if not is_ignored(self.lines, return_stmt.lineno, IGNORE_SUSPICIOUS_TOKEN):
                        self.issues.append(
                            (
                                return_stmt.lineno,
                                return_stmt.col_offset + 1,
                                f"{SUSPICIOUS_RULE_ID} {SUSPICIOUS_RULE_MSG}",
                            )
                        )

        self.generic_visit(node)


def iter_python_files(paths: list[Path]) -> list[Path]:
    results: list[Path] = []

    for path in paths:
        if path.is_file() and path.suffix == ".py":
            results.append(path)
            continue

        if path.is_dir():
            for py_file in path.rglob("*.py"):
                if any(part in {".venv", "venv", "vendor", "node_modules", ".git"} for part in py_file.parts):
                    continue
                results.append(py_file)

    return sorted(set(results))


def check_file(path: Path) -> list[tuple[int, int, str]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    lines = source.splitlines()
    visitor = FallbackVisitor(path, lines)
    visitor.visit(tree)
    return visitor.issues


def main() -> int:
    raw_paths = sys.argv[1:] or ["."]
    paths = [Path(raw) for raw in raw_paths]
    files = iter_python_files(paths)

    findings = 0
    for py_file in files:
        try:
            issues = check_file(py_file)
        except (SyntaxError, UnicodeDecodeError):
            continue

        for line, col, message in issues:
            findings += 1
            print(f"{py_file}:{line}:{col}: {message}")

    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
