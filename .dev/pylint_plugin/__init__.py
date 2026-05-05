"""Agentlinters Pylint plugin.

Provides two checkers that mirror the rules in fallback_checker.py,
but integrated into Pylint via the standard plugin API so they appear
in pylint output alongside all other messages.

Rules
-----
UFB001  provably-unnecessary-fallback
    The left-hand side of an ``or`` expression is a literal that is
    always truthy, so the right-hand side (fallback) is dead code.

SFB001  suspicious-fallback
    An ``except`` handler contains a ``return`` statement while the
    guarded ``try`` body also contains a ``return``.  The error path
    silently recovers to success, which is usually a mistake.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astroid import nodes
from pylint.checkers import BaseChecker

if TYPE_CHECKING:
    from pylint.lint import PyLinter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_definitely_truthy(node: nodes.NodeNG) -> bool:
    """Return True when *node* is a compile-time-truthy literal."""
    if isinstance(node, nodes.Const):
        return bool(node.value)
    if isinstance(node, (nodes.List, nodes.Tuple, nodes.Set)):
        return len(node.elts) > 0
    if isinstance(node, nodes.Dict):
        return len(node.items) > 0
    return False


def _has_return(stmts: list[nodes.NodeNG]) -> bool:
    """Return True when any statement in *stmts* (non-recursively into
    nested functions/classes) is a Return node."""
    stack = list(stmts)
    while stack:
        node = stack.pop()
        if isinstance(node, nodes.Return):
            return True
        for child in node.get_children():
            if isinstance(child, (nodes.FunctionDef, nodes.AsyncFunctionDef, nodes.ClassDef)):
                continue
            stack.append(child)
    return False


def _has_raise(stmts: list[nodes.NodeNG]) -> bool:
    """Return True when any statement in *stmts* is a Raise node."""
    stack = list(stmts)
    while stack:
        node = stack.pop()
        if isinstance(node, nodes.Raise):
            return True
        for child in node.get_children():
            if isinstance(child, (nodes.FunctionDef, nodes.AsyncFunctionDef, nodes.ClassDef)):
                continue
            stack.append(child)
    return False


def _find_returns(stmts: list[nodes.NodeNG]) -> list[nodes.Return]:
    """Collect all Return nodes reachable from *stmts* without crossing
    function or class boundaries."""
    results: list[nodes.Return] = []
    stack = list(stmts)
    while stack:
        node = stack.pop()
        if isinstance(node, nodes.Return):
            results.append(node)
        for child in node.get_children():
            if isinstance(child, (nodes.FunctionDef, nodes.AsyncFunctionDef, nodes.ClassDef)):
                continue
            stack.append(child)
    return results


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------


class FallbackChecker(BaseChecker):
    """Detects provably unnecessary and suspicious fallback patterns."""

    name = "agentlinters-fallback"

    msgs = {
        "W9901": (
            "Provably unnecessary fallback: left side of `or` is always truthy",
            "provably-unnecessary-fallback",
            "The left operand of this `or` expression is a literal that is always "
            "truthy, so the right-hand fallback value is never reached. "
            "Remove the fallback or use a variable instead of a literal.",
        ),
        "W9902": (
            "Suspicious fallback: `except` handler returns success while `try` body also returns",
            "suspicious-fallback",
            "The `except` handler contains a `return` statement while the guarded "
            "`try` block also returns. The error path silently recovers to success, "
            "which is usually a mistake. Consider re-raising or returning a sentinel.",
        ),
    }

    def visit_boolop(self, node: nodes.BoolOp) -> None:
        """Check for provably unnecessary ``or`` fallbacks."""
        if not isinstance(node.op, str) or node.op != "or":
            return
        if len(node.values) < 2:
            return
        left = node.values[0]
        if _is_definitely_truthy(left):
            self.add_message("provably-unnecessary-fallback", node=node)

    def visit_try(self, node: nodes.Try) -> None:
        """Check for suspicious ``except`` recovery patterns."""
        if not _has_return(list(node.body)):
            return
        for handler in node.handlers:
            if _has_raise(list(handler.body)):
                continue
            for return_node in _find_returns(list(handler.body)):
                self.add_message("suspicious-fallback", node=return_node)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(linter: PyLinter) -> None:
    linter.register_checker(FallbackChecker(linter))
