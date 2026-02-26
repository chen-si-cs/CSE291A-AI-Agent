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
    ):
        self.llm_call = llm_call
        self.verbose = verbose
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.messages: List[dict] = []
        self._obs_history: List[dict] = []

    def setup(self, observation: dict) -> None:
        """Initialize conversation with the puzzle context."""
        self.messages = []
        self._obs_history = [observation]

        # Build the initial user message with full puzzle context
        user_msg = self._format_initial_observation(observation)
        self.messages = [
            {"role": "user", "content": user_msg}
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
            msg = last_result.get("message", json.dumps(last_result, default=str))
            budget = observation.get("budget_remaining", "?")
            self.messages.append({
                "role": "user",
                "content": f"Result: {msg}\n\nBudget remaining: {budget} steps.\nWhat's your next action?"
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
        lines.append("Respond with ONE command per turn, e.g.:")
        lines.append("  execute objects(I, True, False, True) -> x1")
        lines.append("  inspect x1")
        lines.append("  submit O")
        lines.append("")
        lines.append("What's your first action?")

        return "\n".join(lines)

    def _parse_response(self, text: str) -> dict:
        """Extract an action command from LLM response text."""
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
        return parse_command(text.strip().split("\n")[-1].strip())


# ── Default system prompt ───────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """\
You are an expert at solving ARC-AGI puzzles using a Domain-Specific Language (DSL).

You are given train input/output pairs and a test input. Your goal is to figure out
the transformation pattern and apply it to the test input using DSL operations.

Available DSL operations include grid transformations (mirror, rotate, crop, fill, paint),
object detection (objects, colorfilter, sizefilter), set operations (merge, difference,
intersection), and higher-order function composition (compose, fork, lbind, rbind, chain).

Strategy:
1. Study the train examples to identify the pattern.
2. Use `execute` to call DSL functions, building up intermediate results.
3. Use `inspect` to examine results when needed.
4. Use `submit` when you have the final grid.

Respond with exactly ONE command per turn. No extra commentary.
"""
