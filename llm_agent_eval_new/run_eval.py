#!/usr/bin/env python3
"""
Evaluate LLM models on ARC puzzles using the oracle DSL environment.

Usage:
    python run_eval.py --model api-gpt-oss-120b
    python run_eval.py --model api-gpt-oss-120b --split test
    python run_eval.py --model all
"""

import argparse
import json, os, re, ast, time, textwrap, requests
from typing import Dict, Any, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Constants ─────────────────────────────────────────────────────
T, F = True, False
ZERO, ONE, TWO, THREE, FOUR, FIVE = 0, 1, 2, 3, 4, 5
SIX, SEVEN, EIGHT, NINE, TEN = 6, 7, 8, 9, 10
NEG_ONE, NEG_TWO = -1, -2
ORIGIN = (0, 0)
UNITY = (1, 1)
NEG_UNITY = (-1, -1)
DOWN, RIGHT, UP, LEFT = (1, 0), (0, 1), (-1, 0), (0, -1)
UP_RIGHT, DOWN_LEFT = (-1, 1), (1, -1)
TWO_BY_TWO = (2, 2)
THREE_BY_THREE = (3, 3)
TWO_BY_ZERO = (2, 0)
ZERO_BY_TWO = (0, 2)

CONSTANTS = {
    "T": T, "F": F,
    "ZERO": ZERO, "ONE": ONE, "TWO": TWO, "THREE": THREE,
    "FOUR": FOUR, "FIVE": FIVE, "SIX": SIX, "SEVEN": SEVEN,
    "EIGHT": EIGHT, "NINE": NINE, "TEN": TEN,
    "NEG_ONE": NEG_ONE, "NEG_TWO": NEG_TWO,
    "ORIGIN": ORIGIN, "UNITY": UNITY, "NEG_UNITY": NEG_UNITY,
    "DOWN": DOWN, "RIGHT": RIGHT, "UP": UP, "LEFT": LEFT,
    "UP_RIGHT": UP_RIGHT, "DOWN_LEFT": DOWN_LEFT,
    "TWO_BY_TWO": TWO_BY_TWO, "THREE_BY_THREE": THREE_BY_THREE,
    "TWO_BY_ZERO": TWO_BY_ZERO, "ZERO_BY_TWO": ZERO_BY_TWO,
}

# ── Puzzle IDs ────────────────────────────────────────────────────
TRAIN_IDS = [
    'c9e6f938', 'a5313dff', '3906de3d', '50cb2852', 'd0f5fe59', '1f876c06',
    '68b16354', '0ca9ddb6', '2dc579da', 'ae4f1146', 'c59eb873', '6fa7a44f',
    'a699fb00', '05f2a901', '5521c0d9', 'a416b8f3', '0b148d64', '007bbfb7',
    '42a50994', '7fe24cdd', 'ac0a08a4', 'a740d043', 'be94b721', '40853293',
    '7b6016b9', '48d8fb45', '5bd6f4ac', 'e9614598', 'c909285e', 'd631b094',
    '3618c87e', '445eab21', '74dd1130', '3aa6fb7a', '39a8645d', '5117e062',
    '4c4377d9', '496994bd', '46f33fce', '363442ee', '44d8ac46', '4347f46a',
    '22eb0ac0', 'c1d99e64', 'b27ca6d3', 'f25fbde4', '4258a5f9', 'd23f8c26',
    '3af2c5a8', 'b94a9452', '543a7ed5', '1e0a9b12', '44f52bb0', 'b60334d2',
    '2dee498d', '67385a82', '8efcae92', '1cf80156', 'ea32f347', '28bf18c6',
    '90c28cc7', 'd2abd087', 'e98196ab', 'dae9d2b5', '6f8cd79b', 'f76d97a5',
    'bb43febb', 'b6afb2da', '46442a0e', 'b1948b0a', '32597951', '662c240a',
    '7b7f7511', '5168d44c', 'c8f0f002', '8d5021e8', '6d0aefbc', '2013d3e2',
    'dc1df850', '9172f3a0', 'f25ffba3', '1b2d62fb', '67e8384a', '7468f01a',
    'd511f180', '56ff96f3', 'ba97ae07', 'f8ff0b80', 'e9afcf9a', 'b9b7f026',
    '5582e5ca', '00d62c1b', '62c24649', 'd5d6de2d', '10fcaaa3', '8be77c9e',
    '67a3c6ac', 'b91ae062', 'dbc1a6ce', 'ce22a75a', '017c7c7b', 'a79310a0',
    'ea786f4a', 'd037b0a7',
]

TEST_IDS = [
    '0520fde7', 'e3497940', '25ff71a9', 'aedd82e4', '08ed6ac7',
    '9f236235', '1c786137', '9ecd008a', '1f85a75f', 'ded97339',
    '3c9b0459', '6150a2bd', 'aabf363d', 'ce9e57f2', 'd10ecb37',
    '41e4d17e', 'c9f8e694', '85c4e7cd', '928ad970', 'ed36ccf7',
    'd406998b', '5614dbcf', 'd9fac9be', '0d3d703e', '9565186b',
    '9dfd6313', '23b5c85d',
]

ALL_MODELS = [
    "api-gpt-oss-120b",
    "api-mistral-small-3.2-2506",
    "minimax.minimax-m2",
    "moonshotai.kimi-k2.5",
    "us.deepseek.r1-v1:0",
]

# ── Config ────────────────────────────────────────────────────────
API_URL = "https://tritonai-api.ucsd.edu/v1/chat/completions"
API_KEY = os.environ.get("TRITONAI_API_KEY", "")
MAX_STEPS = 20
SEED = 42
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
SOLVERS_SOURCE_PATH = os.path.join(SCRIPT_DIR, "solvers.py")
DSL_SOURCE_PATH = os.path.join(SCRIPT_DIR, "dsl.py")

# ── Write arc_types stub ─────────────────────────────────────────
arc_types_path = os.path.join(SCRIPT_DIR, "arc_types.py")
with open(arc_types_path, "w") as f:
    f.write(
        "from typing import Any, Callable, Container, FrozenSet, Tuple as Tuple\n"
        "Boolean = bool\nInteger = int\nNumerical = Any\nIntegerTuple = Tuple\n"
        "IntegerSet = Any\nContainerContainer = Any\nGrid = Any\nCell = Any\n"
        "Object = Any\nObjects = Any\nIndices = Any\nIndicesSet = Any\n"
        "Patch = Any\nElement = Any\nPiece = Any\nTupleTuple = Any\n"
    )

# ── Load DSL ─────────────────────────────────────────────────────
dsl_namespace: Dict[str, Any] = {}
dsl_namespace.update(CONSTANTS)
with open(DSL_SOURCE_PATH) as f:
    dsl_code = f.read()
exec(compile(dsl_code, DSL_SOURCE_PATH, "exec"), dsl_namespace)

DSL_FUNCTIONS: Dict[str, Any] = {
    k: v for k, v in dsl_namespace.items()
    if callable(v) and not k.startswith("_") and k[0].islower()
}
print(f"Loaded {len(DSL_FUNCTIONS)} DSL functions.")


def extract_dsl_docs(source_path):
    with open(source_path) as f:
        source = f.read()
    pattern = r'def (\w+)\((.*?)\)\s*->\s*\w+:\s*"""(.*?)"""'
    matches = re.findall(pattern, source, re.DOTALL)
    docs = {}
    for name, raw_args, docstring in matches:
        arg_lines = [l.strip().rstrip(",") for l in raw_args.split("\n") if l.strip()]
        arg_names = [a.split(":")[0].strip() for a in arg_lines]
        sig = f"{name}({', '.join(arg_names)})"
        docs[name] = f"{sig}  --  {docstring.strip()}"
    return docs


DSL_DOCS = extract_dsl_docs(DSL_SOURCE_PATH)


def parse_solver(puzzle_id, source_path):
    with open(source_path) as f:
        source = f.read()
    pattern = rf'def solve_{puzzle_id}\(I\):\n(.*?)(?=\ndef |\Z)'
    m = re.search(pattern, source, re.DOTALL)
    if not m:
        raise ValueError(f"No solver found for {puzzle_id}")
    body = m.group(1)
    lines = [l.strip() for l in body.strip().split('\n')
             if l.strip() and not l.strip().startswith("return")]
    func_source = f'def solve_{puzzle_id}(I):\n' + body
    tree = ast.parse(textwrap.dedent(func_source))
    dsl_funcs = set()
    constants_used = set()
    all_const_names = set(CONSTANTS.keys())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            dsl_funcs.add(node.func.id)
        if isinstance(node, ast.Name) and node.id in all_const_names:
            constants_used.add(node.id)
    return {
        "lines": lines,
        "dsl_funcs": dsl_funcs,
        "constants_used": constants_used,
        "num_steps": len(lines),
        "gt_solution": lines,
    }


# ── Puzzle / Grid utilities ──────────────────────────────────────
def load_puzzle(puzzle_id, data_dir=DATA_DIR):
    with open(os.path.join(data_dir, f"{puzzle_id}.json")) as f:
        return json.load(f)


def grid_to_tuple(grid_list):
    return tuple(tuple(row) for row in grid_list)


def grid_to_str(grid, indent=2):
    prefix = " " * indent
    return '\n'.join(prefix + str(list(row)) for row in grid)


def grids_match(a, b):
    return tuple(tuple(r) for r in a) == tuple(tuple(r) for r in b)


def count_matching_cells(a, b):
    a_t, b_t = [list(r) for r in a], [list(r) for r in b]
    if len(a_t) != len(b_t) or any(len(ar) != len(br) for ar, br in zip(a_t, b_t)):
        return (0, max(sum(len(r) for r in a_t), sum(len(r) for r in b_t)))
    total = sum(len(r) for r in a_t)
    match = sum(1 for i in range(len(a_t)) for j in range(len(a_t[i])) if a_t[i][j] == b_t[i][j])
    return (match, total)


# ── Prompts ──────────────────────────────────────────────────────
def build_system_prompt():
    return (
        "You are an expert ARC-AGI puzzle solver. You are given a puzzle with "
        "training input-output pairs and a test input grid. Your goal is to "
        "produce the correct test output by applying DSL functions step by step.\n\n"
        "RULES:\n"
        "1. At each turn you see current variables and available DSL functions.\n"
        "2. Output EXACTLY ONE line of Python: `var = func(arg1, arg2, ...)`\n"
        "   Arguments must be existing variable names (I, x1, x2, ...) or constants.\n"
        "3. Name variables x1, x2, x3, ... Use O for the final output assignment.\n"
        "4. Study the training pairs to infer the pattern, then apply DSL functions.\n"
        "5. Output ONLY the code line — no explanation, no backticks, no comments.\n"
        "Reasoning: Low"
    )


def build_initial_observation(puzzle, solver_info):
    parts = ["=== ARC-AGI PUZZLE ===", "TRAINING EXAMPLES:"]
    for i, pair in enumerate(puzzle["train"]):
        parts.append(f"\n--- Train Pair {i+1} ---")
        parts.append(f'Input:\n{grid_to_str(pair["input"])}')
        parts.append(f'Output:\n{grid_to_str(pair["output"])}')
    parts.append("\n--- Test Input ---")
    parts.append(f'Input:\n{grid_to_str(puzzle["test"][0]["input"])}')
    parts.append("\nYour goal: produce the correct output grid for this test input.")
    parts.append("\n=== AVAILABLE DSL FUNCTIONS ===")
    for fname in sorted(solver_info["dsl_funcs"]):
        if fname in DSL_DOCS:
            parts.append(f"  {DSL_DOCS[fname]}")
        else:
            parts.append(f"  {fname}(...)")
    if solver_info["constants_used"]:
        parts.append("\n=== AVAILABLE CONSTANTS ===")
        const_strs = [f'{c}={CONSTANTS[c]}' for c in sorted(solver_info['constants_used'])]
        parts.append("  " + ", ".join(const_strs))
    parts.append("\n=== CURRENT VARIABLES ===")
    parts.append("  I = <test input grid shown above>")
    parts.append("\nWrite the next DSL call:")
    return "\n".join(parts)


# ── DSL Executor ─────────────────────────────────────────────────
class DSLExecutor:
    def __init__(self, test_input_grid, solver_info):
        self.variables = {"I": grid_to_tuple(test_input_grid)}
        self.solver_info = solver_info
        self.exec_ns = {}
        self.exec_ns.update(CONSTANTS)
        self.exec_ns.update(DSL_FUNCTIONS)

    def execute_line(self, line):
        line = line.strip().strip("`").strip()
        if line.startswith("python"):
            line = line[6:].strip()
        if "=" not in line:
            return False, f'Invalid format (no assignment): {line}'
        match = re.match(r'^(\w+)\s*=\s*(.+)$', line)
        if not match:
            return False, f'Could not parse: {line}'
        var_name = match.group(1)
        safe_builtins = {
            'list': list, 'tuple': tuple, 'set': set, 'frozenset': frozenset,
            'int': int, 'float': float, 'str': str, 'bool': bool,
            'len': len, 'range': range, 'enumerate': enumerate, 'zip': zip,
            'min': min, 'max': max, 'sum': sum, 'sorted': sorted,
            'abs': abs, 'round': round,
            'True': True, 'False': False, 'None': None,
            'isinstance': isinstance, 'type': type, 'map': map, 'filter': filter,
        }
        local_ns = {}
        local_ns.update(self.exec_ns)
        local_ns.update(self.variables)

        def _try_exec(ln, builtins_dict, ns):
            exec(ln, {'__builtins__': builtins_dict}, ns)
            return ns.get(var_name)

        try:
            result = _try_exec(line, safe_builtins, local_ns)
        except TypeError:
            def _wrap(fn):
                def wrapper(*a, **kw):
                    r = fn(*a, **kw)
                    return tuple(r) if isinstance(r, frozenset) else r
                return wrapper
            retry_ns = dict(local_ns)
            for k in list(retry_ns):
                if callable(retry_ns[k]) and k in DSL_FUNCTIONS:
                    retry_ns[k] = _wrap(retry_ns[k])
            for k in list(retry_ns):
                if isinstance(retry_ns[k], frozenset):
                    retry_ns[k] = tuple(retry_ns[k])
            try:
                result = _try_exec(line, safe_builtins, retry_ns)
            except Exception as e2:
                return False, f'Execution error: {type(e2).__name__}: {e2}'
        except Exception as e:
            return False, f'Execution error: {type(e).__name__}: {e}'

        if result is not None:
            if isinstance(result, (list, tuple)) and len(result) > 0 and isinstance(result[0], (list, tuple)):
                result = tuple(tuple(row) for row in result)
            self.variables[var_name] = result
            return True, f'OK: {var_name} assigned'
        return False, f'Variable {var_name} not found after execution'

    def get_state_summary(self):
        parts = ["=== CURRENT VARIABLES ==="]
        for name, val in self.variables.items():
            if isinstance(val, tuple) and len(val) > 0 and isinstance(val[0], tuple):
                rows, cols = len(val), len(val[0]) if val else 0
                parts.append(f'  {name} = <{rows}x{cols} grid>')
                parts.append(grid_to_str(val, indent=4))
            else:
                val_str = str(val)
                if len(val_str) > 200:
                    val_str = val_str[:200] + "..."
                parts.append(f'  {name} = {val_str}')
        return "\n".join(parts)

    def has_output(self):
        return "O" in self.variables

    def get_output(self):
        return self.variables.get("O")


# ── ARC Agent ────────────────────────────────────────────────────
class ARCAgent:
    def __init__(self, api_url, api_key, model, seed=42, temperature=0.5, max_tokens=10240):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history = []

    def _api_call(self, messages):
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model, "messages": messages,
            "temperature": self.temperature, "max_tokens": self.max_tokens,
            "max_completion_tokens": self.max_tokens, "seed": self.seed,
        }
        response = requests.post(self.api_url, headers=headers, json=payload, timeout=300)
        return response.json()

    def _call_with_retry(self, messages, max_retries=5, base_sleep=1.0):
        last_response = None
        for attempt in range(max_retries):
            try:
                last_response = self._api_call(messages)
                if "choices" in last_response and "usage" in last_response:
                    return last_response
                # Log unexpected response for debugging
                if last_response:
                    err = last_response.get("error", {})
                    if err:
                        print(f'  API attempt {attempt+1} error: {err}')
            except Exception as e:
                print(f'  API attempt {attempt+1} failed: {e}')
            if attempt < max_retries - 1:
                time.sleep(base_sleep * (2 ** attempt))
        return last_response

    def reset(self, system_prompt):
        self.history = [{"role": "system", "content": system_prompt}]

    def act(self, observation):
        self.history.append({"role": "user", "content": observation})
        response = self._call_with_retry(self.history)
        if response is None:
            action = "O = I"
            stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        else:
            content = (response.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
            action = content.strip() if content else "O = I"
            usage = response.get("usage", {})
            stats = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        action = (action or "").strip().strip("`").strip()
        if action.startswith("python"):
            action = action[6:].strip()
        lines = action.split("\n")
        if len(lines) > 1 and re.match(r'^\w+\s*=\s*\[\[', lines[0]):
            action = " ".join(l.strip() for l in lines)
            depth = 0
            end_idx = None
            for ci, ch in enumerate(action):
                if ch == '[': depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        end_idx = ci
                        break
            if end_idx is not None:
                action = action[:end_idx + 1]
        else:
            action = lines[0].strip().strip("`").strip()
            if action.startswith("python"):
                action = action[6:].strip()
        self.history.append({"role": "assistant", "content": action})
        return action, stats


# ── Game Loop ────────────────────────────────────────────────────
def play_puzzle(agent, puzzle_id, data_dir, max_steps, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{puzzle_id}.log")
    log_file = open(log_path, "w")

    def log(msg=""):
        log_file.write(msg + "\n")
        log_file.flush()
        print(msg)

    puzzle = load_puzzle(puzzle_id, data_dir)
    solver_info = parse_solver(puzzle_id, SOLVERS_SOURCE_PATH)
    test_input = puzzle["test"][0]["input"]
    test_output = puzzle["test"][0]["output"]
    executor = DSLExecutor(test_input, solver_info)
    system_prompt = build_system_prompt()
    agent.reset(system_prompt)
    obs = build_initial_observation(puzzle, solver_info)

    # Record ground truth for analysis
    gt_solution_str = "\n".join(solver_info["gt_solution"])

    trajectory = {
        "puzzle_id": puzzle_id,
        "split": "test" if puzzle_id in TEST_IDS else "train",
        "model": agent.model,
        "max_steps": max_steps,
        "gt_num_steps": solver_info["num_steps"],
        "gt_dsl_funcs": sorted(solver_info["dsl_funcs"]),
        "gt_solution": solver_info["gt_solution"],
        "steps": [],
        "solved": False,
        "final_match": [0, 0],
        "total_tokens": 0,
        "planning_analysis": "",
        "failure_analysis": "",
    }

    log(f'\n{"="*60}')
    log(f'PUZZLE: {puzzle_id}  |  GT steps: {solver_info["num_steps"]}  |  DSL funcs: {solver_info["dsl_funcs"]}')
    log(f'GT solution:')
    for gl in solver_info["gt_solution"]:
        log(f'  {gl}')
    log(f'{"="*60}')

    # Log initial observation (full puzzle context)
    log_file.write("\n--- INITIAL OBSERVATION ---\n")
    log_file.write(obs + "\n")
    log_file.write("--- END INITIAL OBSERVATION ---\n\n")
    log_file.flush()

    # Planning round
    planning_prompt = (
        obs + "\n\nBefore writing any code, please analyze the training input-output pairs "
        "and describe:\n1. What transformation pattern/rule do you see?\n"
        "2. How will you apply the available DSL functions to implement it?\n"
        "3. What is your step-by-step plan?\n\n"
        "Respond with your analysis (free-form text is fine for this round)."
    )
    agent.history.append({"role": "user", "content": planning_prompt})
    planning_response = agent._call_with_retry(agent.history)
    planning_text = ""
    planning_tokens = 0
    if planning_response and "choices" in planning_response:
        planning_text = (planning_response.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or ""
        planning_tokens = planning_response.get("usage", {}).get("total_tokens", 0)
    agent.history.append({"role": "assistant", "content": planning_text})
    trajectory["planning_analysis"] = planning_text
    trajectory["total_tokens"] += planning_tokens

    log("\n--- PLANNING ANALYSIS ---")
    log(planning_text)
    log("--- END PLANNING ---\n")

    obs = (
        "Good. Now begin executing your plan. "
        "Remember: output EXACTLY ONE line of DSL code per turn.\n\n"
        + executor.get_state_summary() + "\n"
        "\nAvailable DSL functions:\n"
        + "\n".join(f"  {DSL_DOCS[fn]}" for fn in sorted(solver_info["dsl_funcs"]) if fn in DSL_DOCS)
        + "\n\nWrite the next DSL call:"
    )

    for step in range(1, max_steps + 1):
        action, stats = agent.act(obs)
        trajectory["total_tokens"] += stats.get("total_tokens", 0)

        log(f'  Step {step:2d} | {action}')

        success, exec_msg = executor.execute_line(action)
        log(f'           -> {exec_msg}')

        solved = False
        match_cells, total_cells = 0, 0
        if executor.has_output():
            output_grid = executor.get_output()
            try:
                match_cells, total_cells = count_matching_cells(output_grid, test_output)
                solved = grids_match(output_grid, test_output)
            except Exception:
                pass

        trajectory["steps"].append({
            "step": step,
            "action": action,
            "success": success,
            "exec_msg": exec_msg,
            "has_output": executor.has_output(),
            "match_cells": match_cells,
            "total_cells": total_cells,
            "solved": solved,
            "tokens": stats.get("total_tokens", 0),
        })

        if solved:
            trajectory["solved"] = True
            trajectory["final_match"] = [match_cells, total_cells]
            log(f'  >>> SOLVED at step {step}! ({match_cells}/{total_cells} cells)')
            break

        # Build feedback observation
        obs_parts = []
        if success:
            obs_parts.append(f"Executed: {action}")
            obs_parts.append(f"Result: {exec_msg}")
        else:
            obs_parts.append(f"ERROR executing: {action}")
            obs_parts.append(f"Error: {exec_msg}")
            obs_parts.append("Please try again with a valid DSL call.")
        obs_parts.append("")
        obs_parts.append(executor.get_state_summary())
        if executor.has_output():
            obs_parts.append(f'\nOutput accuracy: {match_cells}/{total_cells} cells match')
            obs_parts.append("The output is NOT correct yet. Keep refining.")
        obs_parts.append("\nAvailable DSL functions:")
        for fname in sorted(solver_info["dsl_funcs"]):
            if fname in DSL_DOCS:
                obs_parts.append(f"  {DSL_DOCS[fname]}")
        obs_parts.append("\nWrite the next DSL call:")
        obs = "\n".join(obs_parts)

        # Log feedback
        log_file.write(f"    --- feedback (step {step}) ---\n")
        log_file.write(obs + "\n")
        log_file.flush()

    if not trajectory["solved"]:
        if executor.has_output():
            try:
                mc, tc = count_matching_cells(executor.get_output(), test_output)
                trajectory["final_match"] = [mc, tc]
            except Exception:
                pass
        mc, tc = trajectory["final_match"]
        log(f'  >>> NOT solved after {max_steps} steps. Accuracy: {mc}/{tc}')

        # Failure analysis: categorize why it failed
        error_steps = [s for s in trajectory["steps"] if not s["success"]]
        successful_steps = [s for s in trajectory["steps"] if s["success"]]
        if len(error_steps) > len(successful_steps):
            trajectory["failure_analysis"] = f"Execution errors dominated ({len(error_steps)}/{len(trajectory['steps'])} steps failed). LLM struggled with DSL syntax or function usage."
        elif mc > 0 and tc > 0 and mc / tc > 0.8:
            trajectory["failure_analysis"] = f"Close miss: {mc}/{tc} cells correct ({100*mc/tc:.0f}%). LLM understood the pattern but made a minor error."
        elif mc > 0 and tc > 0 and mc / tc > 0.5:
            trajectory["failure_analysis"] = f"Partial understanding: {mc}/{tc} cells correct ({100*mc/tc:.0f}%). LLM grasped some aspects but missed key parts of the transformation."
        else:
            trajectory["failure_analysis"] = f"Poor result: {mc}/{tc} cells correct. LLM likely misunderstood the transformation pattern."

        log(f'  Failure analysis: {trajectory["failure_analysis"]}')

    # Log ground truth comparison
    log(f'\n--- GROUND TRUTH COMPARISON ---')
    log(f'GT solution ({solver_info["num_steps"]} steps):')
    for gl in solver_info["gt_solution"]:
        log(f'  {gl}')
    log(f'Agent solution ({len(trajectory["steps"])} steps):')
    for s in trajectory["steps"]:
        status = "OK" if s["success"] else "ERR"
        log(f'  [{status}] {s["action"]}')
    log(f'--- END COMPARISON ---\n')

    log_file.close()
    return trajectory


# ── Main ─────────────────────────────────────────────────────────
def run_model(model_name, puzzle_ids, split_label):
    """Run evaluation for a single model on given puzzles."""
    # Create model-specific output dirs
    model_safe = model_name.replace("/", "_").replace(":", "_").replace(".", "_")
    log_dir = os.path.join(SCRIPT_DIR, "logs", model_safe)
    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    print(f'\n{"#"*70}')
    print(f'# MODEL: {model_name}')
    print(f'# Puzzles: {len(puzzle_ids)} ({split_label})')
    print(f'# Logs: {log_dir}')
    print(f'{"#"*70}\n')

    agent = ARCAgent(
        api_url=API_URL, api_key=API_KEY, model=model_name,
        seed=SEED, temperature=0.5, max_tokens=10240,
    )

    results = []
    skipped = []
    total_start = time.time()

    for i, pid in enumerate(puzzle_ids):
        puzzle_data = load_puzzle(pid, DATA_DIR)
        test_grid = puzzle_data["test"][0]["input"]
        h, w = len(test_grid), len(test_grid[0]) if test_grid else 0
        if h > 10 or w > 10:
            print(f'[{i+1}/{len(puzzle_ids)}] SKIP {pid}: grid too large ({h}x{w})')
            skipped.append({"puzzle_id": pid, "height": h, "width": w})
            continue

        print(f'\n[{i+1}/{len(puzzle_ids)}] Playing puzzle: {pid}')
        t0 = time.time()
        traj = play_puzzle(agent, pid, DATA_DIR, MAX_STEPS, log_dir)
        traj["time_seconds"] = round(time.time() - t0, 1)
        results.append(traj)

    total_time = round(time.time() - total_start, 1)

    # Summary
    attempted = len(results)
    solved_count = sum(1 for r in results if r["solved"])
    solve_rate = round(100 * solved_count / attempted, 1) if attempted else 0
    total_tokens = sum(r.get("total_tokens", 0) for r in results)

    print(f'\n{"="*70}')
    print(f'  MODEL: {model_name}')
    print(f'  RESULTS SUMMARY')
    print(f'{"="*70}')
    print(f'  Total puzzles:  {len(puzzle_ids)}')
    print(f'  Skipped:        {len(skipped)} (grid > 10x10)')
    print(f'  Attempted:      {attempted}')
    print(f'  Solved:         {solved_count}/{attempted} ({solve_rate}%)')
    print(f'  Total tokens:   {total_tokens:,}')
    print(f'  Total time:     {total_time}s')
    print(f'{"="*70}')

    for r in results:
        mc, tc = r["final_match"]
        status = "SOLVED" if r["solved"] else "FAILED"
        steps_used = len(r["steps"])
        print(f'  {r["puzzle_id"]}: {status} | Steps: {steps_used}/{r["max_steps"]} | '
              f'GT: {r["gt_num_steps"]} | Match: {mc}/{tc} | {r.get("time_seconds",0)}s')

    # Save detailed results JSON
    output_path = os.path.join(results_dir, f"{model_safe}_{split_label}.json")
    save_data = {
        "model": model_name,
        "split": split_label,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_puzzles": len(puzzle_ids),
        "skipped": len(skipped),
        "attempted": attempted,
        "solved": solved_count,
        "solve_rate": solve_rate,
        "total_tokens": total_tokens,
        "total_time_seconds": total_time,
        "skipped_puzzles": skipped,
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f'Results saved to {output_path}')

    return save_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM on ARC puzzles with oracle DSL")
    parser.add_argument("--model", type=str, default="api-gpt-oss-120b",
                        help=f"Model name or 'all'. Available: {ALL_MODELS}")
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"],
                        help="Which puzzle split to evaluate")
    args = parser.parse_args()

    # Select puzzles
    if args.split == "train":
        puzzle_ids = TRAIN_IDS
        split_label = "train"
    elif args.split == "test":
        puzzle_ids = TEST_IDS
        split_label = "test"
    else:
        puzzle_ids = TRAIN_IDS + TEST_IDS
        split_label = "all"

    # Select models
    if args.model == "all":
        models = ALL_MODELS
    else:
        models = [args.model]

    all_summaries = []
    for model in models:
        summary = run_model(model, puzzle_ids, split_label)
        all_summaries.append(summary)

    # Print cross-model comparison if multiple models
    if len(all_summaries) > 1:
        print(f'\n{"="*70}')
        print("CROSS-MODEL COMPARISON")
        print(f'{"="*70}')
        print(f'{"Model":<35} {"Solved":>8} {"Rate":>8} {"Tokens":>12} {"Time":>8}')
        print("-" * 75)
        for s in all_summaries:
            print(f'{s["model"]:<35} {s["solved"]}/{s["attempted"]:>5} {s["solve_rate"]:>7.1f}% {s["total_tokens"]:>11,} {s["total_time_seconds"]:>7.0f}s')
