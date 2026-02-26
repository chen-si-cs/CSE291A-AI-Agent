"""
PuzzleDB: loads and indexes ARC-AGI puzzle JSON files.

Expected directory layout:
    data/
      train/
        00d62c1b.json
        ...
      evaluation/
        ...

Each JSON has:
    {
      "train": [{"input": [[...]], "output": [[...]]}, ...],
      "test":  [{"input": [[...]], "output": [[...]]}]
    }
"""

from __future__ import annotations
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class Puzzle:
    """One ARC puzzle with train pairs and test pairs."""

    def __init__(self, puzzle_id: str, data: dict):
        self.puzzle_id = puzzle_id
        self.train: List[dict] = data["train"]
        self.test: List[dict] = data["test"]

    @property
    def num_train(self) -> int:
        return len(self.train)

    @property
    def num_test(self) -> int:
        return len(self.test)

    def train_input(self, idx: int = 0) -> tuple:
        return _to_grid(self.train[idx]["input"])

    def train_output(self, idx: int = 0) -> tuple:
        return _to_grid(self.train[idx]["output"])

    def test_input(self, idx: int = 0) -> tuple:
        return _to_grid(self.test[idx]["input"])

    def test_output(self, idx: int = 0) -> tuple:
        return _to_grid(self.test[idx]["output"])

    def __repr__(self):
        return (f"Puzzle({self.puzzle_id}, "
                f"train={self.num_train}, test={self.num_test})")


class PuzzleDB:
    """Index of all available ARC puzzles."""

    def __init__(self, data_dirs: Optional[List[str]] = None):
        self._puzzles: Dict[str, Puzzle] = {}
        if data_dirs:
            for d in data_dirs:
                self.load_dir(d)

    def load_dir(self, dir_path: str):
        """Load all .json puzzle files from a directory."""
        p = Path(dir_path)
        if not p.exists():
            return
        for f in sorted(p.glob("*.json")):
            puzzle_id = f.stem
            try:
                with open(f) as fh:
                    data = json.load(fh)
                self._puzzles[puzzle_id] = Puzzle(puzzle_id, data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: skipping {f}: {e}")

    def load_single(self, json_path: str) -> Puzzle:
        """Load one puzzle file and add it to the DB."""
        p = Path(json_path)
        puzzle_id = p.stem
        with open(p) as fh:
            data = json.load(fh)
        puzzle = Puzzle(puzzle_id, data)
        self._puzzles[puzzle_id] = puzzle
        return puzzle

    def get(self, puzzle_id: str) -> Puzzle:
        if puzzle_id not in self._puzzles:
            raise KeyError(f"Puzzle '{puzzle_id}' not found. "
                           f"Available: {len(self._puzzles)} puzzles")
        return self._puzzles[puzzle_id]

    def random(self) -> Puzzle:
        return random.choice(list(self._puzzles.values()))

    def ids(self) -> List[str]:
        return sorted(self._puzzles.keys())

    def __len__(self):
        return len(self._puzzles)

    def __contains__(self, puzzle_id):
        return puzzle_id in self._puzzles

    def __repr__(self):
        return f"PuzzleDB({len(self._puzzles)} puzzles)"


def _to_grid(grid_list) -> tuple:
    return tuple(tuple(row) for row in grid_list)
