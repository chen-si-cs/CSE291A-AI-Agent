"""
Extract env action sequences from solver functions in solvers.py.

Each solver is a function solve_<puzzle_id>(I) that does:
  x1 = dsl_func(...); x2 = ...; O = ...; return O
We parse the AST to get (function_name, args, store_as) and convert to
env action dicts: execute / submit.
"""

from __future__ import annotations
import ast
import inspect
from typing import Any, Dict, List, Optional

from constants import CONSTANT_REGISTRY


def _get_dsl_function_names() -> set:
    """Names that are DSL functions (for resolving Call args)."""
    try:
        from dsl_engine import DSLEngine
        return set(DSLEngine().function_names())
    except Exception:
        return set()


# Lazy init
_DSL_NAMES: Optional[set] = None


def _is_constant(name: str) -> bool:
    return name in CONSTANT_REGISTRY


def _is_dsl_function(name: str) -> bool:
    global _DSL_NAMES
    if _DSL_NAMES is None:
        _DSL_NAMES = _get_dsl_function_names()
    return name in _DSL_NAMES


def _arg_to_action_arg(node: ast.AST) -> Any:
    """Convert an AST argument to env action arg: string (inventory/constant/func) or literal."""
    if isinstance(node, ast.Name):
        n = node.id
        if _is_constant(n):
            return CONSTANT_REGISTRY[n]  # literal
        # I, x1, x2, O, or DSL function ref (e.g. size)
        return n
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        elts = [_arg_to_action_arg(e) for e in node.elts]
        return tuple(elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if isinstance(node.operand, ast.Constant):
            return -node.operand.value
        if isinstance(node.operand, ast.Name) and node.operand.id in ("ONE", "TWO"):
            return -CONSTANT_REGISTRY.get(node.operand.id, 0)
    return None


def _call_to_action(call: ast.Call, store_as: str) -> Optional[dict]:
    """Convert AST Call node to env execute action dict."""
    func_name = None
    if isinstance(call.func, ast.Name):
        func_name = call.func.id
    elif isinstance(call.func, ast.Attribute):
        func_name = call.func.attr
    if not func_name:
        return None
    args = []
    for arg in call.args:
        a = _arg_to_action_arg(arg)
        if a is None:
            return None
        args.append(a)
    return {
        "type": "execute",
        "function": func_name,
        "args": args,
        "store_as": store_as,
    }


def solver_function_to_actions(solve_fn: callable) -> List[dict]:
    """
    Parse a solver function and return a list of env action dicts.

    Returns:
        [{"type": "execute", "function": "...", "args": [...], "store_as": "x1"}, ..., {"type": "submit", "answer": "O"}]
    """
    try:
        src = inspect.getsource(solve_fn)
    except OSError:
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    actions = []
    return_var = None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    # single target: x1 = call(...)
                    if len(stmt.targets) != 1:
                        continue
                    target = stmt.targets[0]
                    if not isinstance(target, ast.Name):
                        continue
                    store_as = target.id
                    if isinstance(stmt.value, ast.Call):
                        act = _call_to_action(stmt.value, store_as)
                        if act:
                            actions.append(act)
                    # skip other value types (e.g. Compare for branch)
                elif isinstance(stmt, ast.Return) and stmt.value is not None:
                    if isinstance(stmt.value, ast.Name):
                        return_var = stmt.value.id
                    break
            break

    if return_var and actions:
        actions.append({"type": "submit", "answer": return_var})
    return actions


def get_solver_puzzle_id(solve_fn: callable) -> Optional[str]:
    """Extract puzzle_id from solve_<puzzle_id>."""
    name = getattr(solve_fn, "__name__", "")
    if name.startswith("solve_"):
        return name[6:]  # strip "solve_"
    return None
