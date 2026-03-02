"""
LLMAgent: uses a language model API to decide actions.

Constructs a prompt from the observation and parses the LLM's text
response into an action dict.
"""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Callable

from agents.base_agent import BaseAgent
from text_parser import parse_command


class LLMAgent(BaseAgent):
    """
    Agent that uses an LLM to generate text commands.

    Requires an `llm_call` function:
        def llm_call(messages: list[dict]) -> str:
            # calls your LLM API and returns the assistant's text response
            ...

    Example with Anthropic:
        import anthropic
        client = anthropic.Anthropic()
        def call(messages):
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=messages,
            )
            return resp.content[0].text
        agent = LLMAgent(llm_call=call)
    """

    def __init__(
        self,
        llm_call: Optional[Callable] = None,
        system_prompt: Optional[str] = None,
        verbose: bool = False,
        use_system_message: bool = True,
    ):
        self.llm_call = llm_call
        self.verbose = verbose
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.use_system_message = use_system_message
        self.messages: List[dict] = []
        self._obs_history: List[dict] = []

    def setup(self, observation: dict) -> None:
        """Initialize conversation with the puzzle context."""
        self.messages = []
        self._obs_history = [observation]

        # Build the initial user message with full puzzle context
        user_msg = self._format_initial_observation(observation)
        if self.use_system_message:
            self.messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg},
            ]
        else:
            self.messages = [
                {"role": "user", "content": self.system_prompt + "\n\n" + user_msg}
            ]

    def act(self, observation: dict) -> dict:
        """
        Ask the LLM what to do next, parse its response into an action.
        """
        if not self.llm_call:
            raise RuntimeError(
                "LLMAgent requires an llm_call function. "
                "Pass it in the constructor: LLMAgent(llm_call=my_func)"
            )

        # Add observation context if not the first turn
        if len(self._obs_history) > 1:
            last_result = observation.get("last_action_result", {})
            msg = last_result.get("message", "")
            err = last_result.get("error", "")
            budget = observation.get("budget_remaining", "?")

            # Build inventory summary
            inv = observation.get("inventory", {})
            inv_lines = [f"  {k}: {v.get('type','?')} — {v.get('preview','')}"
                         for k, v in inv.items()]
            inv_str = "\n".join(inv_lines) if inv_lines else "  (empty)"

            feedback = f"Result: {err or msg}\n\nInventory:\n{inv_str}\n\nBudget remaining: {budget} steps.\nRespond with your next command (one line, no explanation):"
            self.messages.append({
                "role": "user",
                "content": feedback
            })

        self._obs_history.append(observation)

        # Call the LLM
        response_text = self.llm_call(self.messages)

        if self.verbose:
            print(f"[LLM] {response_text}")

        # Add to conversation
        self.messages.append({"role": "assistant", "content": response_text})

        # Parse the response into an action
        action = self._parse_response(response_text)
        return action

    def on_step_result(self, observation: dict, reward: float,
                       done: bool, info: dict) -> None:
        pass  # handled in act() via observation

    def on_episode_end(self, total_reward: float, steps: int, info: dict) -> None:
        if self.verbose:
            status = "SOLVED" if info.get("success") else "FAILED"
            print(f"[LLM Agent] Episode ended: {status}, "
                  f"reward={total_reward:.3f}, steps={steps}")

    def _format_initial_observation(self, obs: dict) -> str:
        """Format the initial observation as a prompt."""
        lines = [f"Puzzle: {obs['puzzle_id']}",
                 f"Budget: {obs['budget_remaining']} steps", ""]

        # Show train examples
        train_examples = obs.get("train_examples", [])
        for ex in train_examples:
            lines.append(f"--- Train Example {ex['index']} ---")
            lines.append(f"Input ({ex['input_shape']}):")
            lines.append(ex["input_grid"])
            lines.append(f"Output ({ex['output_shape']}):")
            lines.append(ex["output_grid"])
            lines.append(f"Changes: {ex['diff']}")
            lines.append("")

        # Show test input
        test = obs.get("test_input", {})
        lines.append(f"--- Test Input ({test.get('shape', '?')}) ---")
        lines.append(test.get("grid", ""))
        lines.append("")

        lines.append("Your inventory starts with variable 'I' = test input grid.")
        lines.append("Use DSL commands to transform it into the expected output.")
        lines.append("")
        lines.append("IMPORTANT: Respond with ONLY ONE command, nothing else. No explanation, no reasoning.")
        lines.append("Example commands:")
        lines.append("  execute objects(I, True, False, False) -> x1")
        lines.append("  execute colorfilter(x1, 0) -> x2")
        lines.append("  inspect x1")
        lines.append("  submit O")
        lines.append("")
        lines.append("Your first command:")

        return "\n".join(lines)

    def _parse_response(self, text: str) -> dict:
        """Extract an action command from LLM response text."""
        if text is None:
            return {"type": "error", "message": "LLM returned empty response"}

        # Look for a line that starts with a command keyword
        for line in text.strip().split("\n"):
            line = line.strip()
            # Skip empty lines, comments, markdown
            if not line or line.startswith("#") or line.startswith("```"):
                continue
            # Remove leading markers like ">" or "-" or "*"
            if line.startswith(">") or line.startswith("-") or line.startswith("*"):
                line = line[1:].strip()

            action = parse_command(line)
            if action["type"] != "error":
                return action

        # Fallback: try the whole text
        last_line = text.strip().split("\n")[-1].strip()
        return parse_command(last_line)


# ── Default system prompt ───────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """\
You are an expert at solving ARC-AGI puzzles using a Domain-Specific Language (DSL).

You are given train input/output pairs and a test input. Your goal is to figure out
the transformation pattern and apply it to the test input using DSL operations.

## Commands (respond with EXACTLY ONE per turn, no extra text)

  execute FUNC(arg1, arg2, ...) -> varname
  inspect varname
  list_functions filter_keyword
  submit varname

String args that match an inventory variable name are resolved to that variable's value.
Literal values: True, False, integers (0-9), tuples like (1,2).

## Key DSL Functions

Grid transforms:
  hmirror(grid) / vmirror(grid) / dmirror(grid) / cmirror(grid) — mirror horizontally/vertically/diagonally/counter-diag
  rot90(grid) / rot180(grid) / rot270(grid) — rotate
  tophalf(grid) / bottomhalf(grid) / lefthalf(grid) / righthalf(grid) — halves
  trim(grid) — remove border rows/cols of background color
  crop(grid, start, dims) — crop subgrid
  upscale(grid, factor) — scale up by integer factor
  hconcat(a, b) / vconcat(a, b) — concatenate grids
  canvas(color, (h,w)) — create solid grid
  fill(grid, color, indices) — fill specific cells with a color
  paint(grid, obj) — paint an object onto a grid
  underfill(grid, color, indices) — fill only background cells
  cellwise(a, b, fallback) — cellwise combination of two grids
  compress(grid) — remove frontiers

Object detection:
  objects(grid, univalued, diagonal, without_bg) — find connected components
    e.g., objects(I, True, False, False) finds all single-color connected regions
  colorfilter(objs, color) — keep objects of specific color
  sizefilter(objs, n) — keep objects of specific size
  palette(obj_or_grid) — set of colors used
  color(obj) — the color of a single-color object
  shape(grid_or_patch) — (height, width)
  mostcolor(grid) / leastcolor(grid) — most/least common color
  ofcolor(grid, color) — indices of cells with given color

Index/set operations:
  toindices(obj) — get cell positions from object
  asindices(grid) — all cell positions
  fill(grid, color, indices) — fill cells at indices
  combine(a, b) / intersection(a, b) / difference(a, b) — set ops
  merge(objs) — merge all objects into one
  mfilter(objs, func) — filter & merge objects matching predicate
  sfilter(container, func) — filter keeping items matching predicate
  extract(objs, func) — filter & return matching objects

Higher-order (critical for complex puzzles):
  compose(outer, inner) — compose two functions: outer(inner(x))
  fork(outer, f, g) — fork(o,f,g)(x) = o(f(x), g(x))
  lbind(func, arg) — partial: lbind(f, a)(x) = f(a, x)
  rbind(func, arg) — partial: rbind(f, a)(x) = f(x, a)
  flip(func) — negate a boolean function
  chain(h, g, f) — h(g(f(x)))
  apply(func, container) — map func over container
  argmax(container, func) / argmin(container, func)
  matcher(func, target) — creates predicate: matcher(color, 3)(obj) = (color(obj)==3)
  rbind(bordering, I) — creates func that checks if patch borders grid edge

## Worked Example (puzzle 00d62c1b: fill enclosed regions with color 4)

  execute objects(I, True, False, False) -> x1     # find all connected regions
  execute colorfilter(x1, 0) -> x2                 # keep only background (black) regions
  execute rbind(bordering, I) -> x3                # func: does patch touch grid border?
  execute compose(flip, x3) -> x4                  # func: does patch NOT touch border?
  execute mfilter(x2, x4) -> x5                    # get enclosed background regions
  execute fill(I, 4, x5) -> O                      # fill them with color 4
  submit O

## Strategy
1. Study train examples carefully — what changes between input and output?
2. Use objects() to decompose the grid into parts.
3. Use higher-order functions (compose, rbind, lbind, flip) to build predicates.
4. Use fill/paint to construct the output. Name your final result "O".
5. submit O when ready.
"""
