"""
DSLEngine: wraps all DSL functions with argument resolution, type checking,
and safe execution.

Key responsibilities:
  - Registry of all DSL functions with signatures and docstrings
  - Resolve action arguments: inventory refs, named constants, DSL function refs, literals
  - Execute a DSL call safely (catching errors, timeouts)
  - Provide function metadata for the agent
"""

from __future__ import annotations
import inspect
import signal
import traceback
from typing import Any, Dict, List, Optional, Tuple
from functools import wraps

# Import the actual DSL module
import dsl as _dsl_module
from constants import CONSTANT_REGISTRY


# ── Timeout helper ──────────────────────────────────────────────────────

class DSLTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise DSLTimeout("DSL function execution timed out")


# ── DSL Function metadata ──────────────────────────────────────────────

class DSLFuncInfo:
    """Metadata for one DSL function."""
    def __init__(self, name: str, func: callable):
        self.name = name
        self.func = func
        sig = inspect.signature(func)
        self.params = list(sig.parameters.keys())
        self.n_args = len(self.params)
        self.signature_str = str(sig)
        self.doc = (func.__doc__ or "").strip()

    def __repr__(self):
        return f"DSLFunc({self.name}{self.signature_str})"


# ── The Engine ──────────────────────────────────────────────────────────

class DSLEngine:
    """
    Wraps the DSL for safe, agent-driven execution.

    Usage:
        engine = DSLEngine()
        result = engine.execute("objects", [grid, True, False, True])
    """

    def __init__(self, timeout_sec: float = 5.0):
        self.timeout_sec = timeout_sec
        self._functions: Dict[str, DSLFuncInfo] = {}
        self._load_functions()

    def _load_functions(self):
        """Discover all public functions in dsl.py."""
        for name in dir(_dsl_module):
            if name.startswith("_"):
                continue
            obj = getattr(_dsl_module, name)
            if callable(obj) and not isinstance(obj, type):
                # Skip imported builtins / types
                try:
                    mod = getattr(obj, '__module__', '')
                    if 'dsl' in str(mod) or mod == '' or mod is None:
                        info = DSLFuncInfo(name, obj)
                        self._functions[name] = info
                except Exception:
                    pass

        # Ensure we have the core functions even if module detection failed
        _ensure = [
            'identity', 'add', 'subtract', 'multiply', 'divide',
            'objects', 'fill', 'paint', 'ofcolor', 'colorfilter',
            'compose', 'fork', 'lbind', 'rbind', 'chain',
            'apply', 'mapply', 'sfilter', 'mfilter', 'merge',
            'shift', 'normalize', 'recolor', 'canvas', 'crop',
            'hmirror', 'vmirror', 'dmirror', 'cmirror',
            'rot90', 'rot180', 'rot270',
            'hconcat', 'vconcat', 'replace', 'switch',
            'height', 'width', 'shape', 'size', 'color',
            'palette', 'numcolors', 'mostcolor', 'leastcolor',
            'uppermost', 'lowermost', 'leftmost', 'rightmost',
            'ulcorner', 'urcorner', 'llcorner', 'lrcorner',
            'backdrop', 'delta', 'inbox', 'outbox', 'box',
            'toindices', 'asobject', 'toobject', 'subgrid',
            'frontiers', 'compress', 'trim', 'cover', 'move',
            'neighbors', 'dneighbors', 'ineighbors',
            'adjacent', 'bordering', 'manhattan',
            'hmatching', 'vmatching', 'connect', 'shoot',
            'partition', 'fgpartition', 'asindices',
            'tophalf', 'bottomhalf', 'lefthalf', 'righthalf',
            'hsplit', 'vsplit', 'cellwise', 'underfill', 'underpaint',
            'upscale', 'downscale', 'hupscale', 'vupscale',
            'occurrences', 'gravitate', 'position', 'center', 'centerofmass',
            'corners', 'index', 'colorcount', 'sizefilter',
            'matcher', 'extract', 'argmax', 'argmin',
            'valmax', 'valmin', 'maximum', 'minimum',
            'mostcommon', 'leastcommon',
            'first', 'last', 'insert', 'remove', 'other',
            'combine', 'intersection', 'difference', 'dedupe',
            'order', 'repeat', 'interval', 'pair', 'product',
            'totuple', 'initset', 'astuple',
            'branch', 'power', 'rapply', 'papply', 'mpapply', 'prapply',
            'even', 'double', 'halve', 'flip', 'invert', 'sign',
            'positive', 'increment', 'decrement', 'crement',
            'greater', 'equality', 'contained', 'both', 'either',
            'toivec', 'tojvec',
            'portrait', 'square', 'vline', 'hline',
            'hfrontier', 'vfrontier',
            'hperiod', 'vperiod',
        ]
        for name in _ensure:
            if name not in self._functions:
                func = getattr(_dsl_module, name, None)
                if func and callable(func):
                    self._functions[name] = DSLFuncInfo(name, func)

    # ── public API ──────────────────────────────────────────────────

    def has_function(self, name: str) -> bool:
        return name in self._functions

    def get_function(self, name: str) -> DSLFuncInfo:
        if name not in self._functions:
            raise KeyError(f"Unknown DSL function: '{name}'")
        return self._functions[name]

    def get_function_ref(self, name: str) -> callable:
        """Return the raw callable for a DSL function (for passing as arg)."""
        return self.get_function(name).func

    def function_names(self) -> List[str]:
        return sorted(self._functions.keys())

    def function_catalog(self, filter_str: str = "") -> Dict[str, dict]:
        """Return metadata dict for display to agent."""
        out = {}
        for name, info in sorted(self._functions.items()):
            if filter_str and filter_str.lower() not in name.lower():
                continue
            out[name] = {
                "signature": info.signature_str,
                "doc": info.doc,
                "n_args": info.n_args,
                "params": info.params,
            }
        return out

    def resolve_args(self, raw_args: list, inventory, allow_funcrefs: bool = True) -> list:
        """
        Resolve a list of raw action arguments into actual Python values.

        Resolution order for each arg:
          1. If it's already a non-string Python value (int, bool, tuple, etc.) → use as-is
          2. If it's a string matching an inventory variable name → inventory.get(name)
          3. If it's a string matching a named constant → constant value
          4. If it's a string matching a DSL function name → function reference
          5. Otherwise → error
        """
        resolved = []
        for arg in raw_args:
            resolved.append(self._resolve_one(arg, inventory, allow_funcrefs))
        return resolved

    def _resolve_one(self, arg: Any, inventory, allow_funcrefs: bool) -> Any:
        """Resolve a single argument."""
        # Already a concrete value (not a string)
        if isinstance(arg, (int, float, bool, tuple, list, frozenset, set)):
            return arg
        if callable(arg) and not isinstance(arg, str):
            return arg
        if arg is None:
            return arg

        if not isinstance(arg, str):
            return arg

        # String-based resolution
        name = arg.strip()

        # 1. Inventory variable
        if inventory.has(name):
            return inventory.get(name)

        # 2. Named constant
        if name in CONSTANT_REGISTRY:
            return CONSTANT_REGISTRY[name]

        # 3. Boolean literals
        if name in ("True", "true"):
            return True
        if name in ("False", "false"):
            return False

        # 4. Integer literal
        try:
            return int(name)
        except ValueError:
            pass

        # 5. DSL function reference
        if allow_funcrefs and name in self._functions:
            return self._functions[name].func

        raise ValueError(
            f"Cannot resolve argument '{name}': "
            f"not an inventory variable, constant, or DSL function"
        )

    def execute(self, func_name: str, resolved_args: list) -> Any:
        """
        Execute a DSL function with already-resolved arguments.
        Returns the result. Raises on error/timeout.
        """
        info = self.get_function(func_name)

        # Set timeout (only works on Unix)
        use_alarm = hasattr(signal, 'SIGALRM')
        if use_alarm:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(int(self.timeout_sec) or 1)
        try:
            result = info.func(*resolved_args)
        except DSLTimeout:
            raise
        except Exception as e:
            raise DSLExecutionError(
                f"Error executing {func_name}({_fmt_args(resolved_args)}): "
                f"{type(e).__name__}: {e}"
            ) from e
        finally:
            if use_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        return result


class DSLExecutionError(Exception):
    """Raised when a DSL function call fails."""
    pass


def _fmt_args(args: list, max_repr: int = 60) -> str:
    """Format args for error messages."""
    parts = []
    for a in args:
        r = repr(a)
        if len(r) > max_repr:
            r = r[:max_repr] + "..."
        parts.append(r)
    return ", ".join(parts)
