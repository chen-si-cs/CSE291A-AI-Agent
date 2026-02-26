"""
TextParser: converts natural-language-like text commands into action dicts.

Supported command formats:

    execute objects(I, True, False, True) -> x1
    execute objects(I, True, False, True)           # auto-name result
    inspect x1
    submit O
    undo
    reset
    list_functions
    list_functions color
    train 0 input
    train 0 output
    train 0 diff
    help
    inventory
"""

from __future__ import annotations
import re
import ast
from typing import Any, Dict, List, Optional, Tuple


def parse_command(text: str) -> Dict[str, Any]:
    """
    Parse a text command into an action dict.

    Returns a dict with at minimum {"type": <action_type>}.
    On parse failure, returns {"type": "error", "message": "..."}.
    """
    text = text.strip()
    if not text:
        return {"type": "error", "message": "Empty command"}

    # Split into first word and rest
    parts = text.split(None, 1)
    cmd = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    if cmd in ("execute", "exec", "call", "run", "do"):
        return _parse_execute(rest)
    elif cmd in ("inspect", "look", "view", "show"):
        return _parse_inspect(rest)
    elif cmd in ("submit", "answer"):
        return _parse_submit(rest)
    elif cmd == "undo":
        return {"type": "undo"}
    elif cmd in ("reset", "clear"):
        return {"type": "reset_inventory"}
    elif cmd in ("list_functions", "functions", "funcs", "tools"):
        return {"type": "list_functions", "filter": rest.strip()}
    elif cmd in ("train", "example"):
        return _parse_train_inspect(rest)
    elif cmd in ("help", "?"):
        return {"type": "help"}
    elif cmd in ("inventory", "inv", "vars"):
        return {"type": "inventory"}
    else:
        # Try to parse as a direct execute: "objects(I, True, False, True) -> x1"
        result = _parse_execute(text)
        if result["type"] != "error":
            return result
        return {"type": "error", "message": f"Unknown command: '{cmd}'"}


def _parse_execute(text: str) -> Dict[str, Any]:
    """
    Parse: func_name(arg1, arg2, ...) -> store_name
    or:    func_name(arg1, arg2, ...)
    """
    text = text.strip()
    if not text:
        return {"type": "error", "message": "Execute requires a function call"}

    # Extract store_as from "-> name"
    store_as = None
    arrow_match = re.search(r'\s*->\s*(\w+)\s*$', text)
    if arrow_match:
        store_as = arrow_match.group(1)
        text = text[:arrow_match.start()].strip()

    # Parse "func_name(arg1, arg2, ...)"
    match = re.match(r'^(\w+)\s*\((.*)?\)\s*$', text, re.DOTALL)
    if not match:
        return {"type": "error",
                "message": f"Cannot parse function call: '{text}'.\n"
                           f"Expected: func_name(arg1, arg2, ...) [-> var_name]"}

    func_name = match.group(1)
    args_str = match.group(2) or ""

    # Parse the argument list
    args = _parse_args(args_str)

    action = {
        "type": "execute",
        "function": func_name,
        "args": args,
    }
    if store_as:
        action["store_as"] = store_as
    return action


def _parse_args(args_str: str) -> list:
    """
    Parse a comma-separated argument string into a list of values.

    Handles: integers, booleans, strings (variable refs), tuples.
    """
    args_str = args_str.strip()
    if not args_str:
        return []

    # Use a state machine to split by commas, respecting parens
    args = []
    depth = 0
    current = []
    for ch in args_str:
        if ch == '(' or ch == '[':
            depth += 1
            current.append(ch)
        elif ch == ')' or ch == ']':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        args.append("".join(current).strip())

    return [_parse_single_arg(a) for a in args if a]


def _parse_single_arg(s: str) -> Any:
    """
    Parse a single argument token.

    Returns:
      - bool for True/False
      - int for integer literals
      - tuple for (a, b) syntax
      - str for everything else (inventory refs, constant names, function names)
    """
    s = s.strip()

    # Boolean
    if s in ("True", "T"):
        return True
    if s in ("False", "F"):
        return False

    # Try integer
    try:
        return int(s)
    except ValueError:
        pass

    # Try tuple literal like (1, 2) or (-1, 0)
    if s.startswith("(") and s.endswith(")"):
        try:
            val = ast.literal_eval(s)
            if isinstance(val, tuple):
                return val
        except (ValueError, SyntaxError):
            pass

    # Otherwise keep as string (will be resolved by engine)
    return s


def _parse_inspect(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {"type": "error", "message": "Inspect requires a variable name"}
    return {"type": "inspect", "target": text.split()[0]}


def _parse_submit(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {"type": "error", "message": "Submit requires a variable name"}
    return {"type": "submit", "answer": text.split()[0]}


def _parse_train_inspect(text: str) -> Dict[str, Any]:
    """Parse: <index> [input|output|diff]"""
    parts = text.strip().split()
    if not parts:
        return {"type": "train_inspect", "index": 0, "which": "both"}
    try:
        index = int(parts[0])
    except ValueError:
        return {"type": "error",
                "message": f"Train inspect: expected integer index, got '{parts[0]}'"}
    which = parts[1] if len(parts) > 1 else "both"
    if which not in ("input", "output", "diff", "both"):
        which = "both"
    return {"type": "train_inspect", "index": index, "which": which}


# ── Help text ───────────────────────────────────────────────────────────

HELP_TEXT = """\
╔══════════════════════════════════════════════════════════╗
║                  ARC DSL SIMULATOR — HELP                ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  EXECUTE a DSL function:                                 ║
║    execute func(arg1, arg2, ...) -> var_name             ║
║    e.g.: execute objects(I, True, False, True) -> x1     ║
║          execute fill(I, 4, x3) -> O                     ║
║                                                          ║
║  INSPECT an inventory variable:                          ║
║    inspect x1                                            ║
║                                                          ║
║  SUBMIT your answer:                                     ║
║    submit O                                              ║
║                                                          ║
║  VIEW training examples:                                 ║
║    train 0            (show example 0)                   ║
║    train 1 diff       (show changes for example 1)       ║
║                                                          ║
║  INVENTORY / MANAGEMENT:                                 ║
║    inventory           (list all variables)              ║
║    undo                (remove last variable)            ║
║    reset               (clear all except I)              ║
║                                                          ║
║  LIST DSL FUNCTIONS:                                     ║
║    functions            (show all)                       ║
║    functions color      (filter by substring)            ║
║                                                          ║
║  Arguments can be:                                       ║
║    - Inventory variables: I, x1, x2, ...                 ║
║    - Named constants: ZERO, ONE, ..., T, F, ORIGIN, ...  ║
║    - DSL function refs: flip, size, identity, ...        ║
║    - Literal ints: 0, 1, 4, -1                           ║
║    - Literal tuples: (1, 0), (-1, -1)                    ║
║    - Booleans: True, False                               ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""
