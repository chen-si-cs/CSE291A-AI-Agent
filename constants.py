"""
Named constants used by the DSL solvers.

These match the identifiers imported via `from constants import *` in solvers.py.
"""

# Integer color / count constants
ZERO = 0
ONE = 1
TWO = 2
THREE = 3
FOUR = 4
FIVE = 5
SIX = 6
SEVEN = 7
EIGHT = 8
NINE = 9
TEN = 10

NEG_ONE = -1
NEG_TWO = -2

# Boolean
T = True
F = False

# Directional tuples  (row_delta, col_delta)
ORIGIN = (0, 0)
UNITY = (1, 1)
NEG_UNITY = (-1, -1)

DOWN = (1, 0)
UP = (-1, 0)
RIGHT = (0, 1)
LEFT = (0, -1)
DOWN_LEFT = (1, -1)
UP_RIGHT = (-1, 1)

# Common dimension tuples
TWO_BY_TWO = (2, 2)
THREE_BY_THREE = (3, 3)
TWO_BY_ZERO = (2, 0)
ZERO_BY_TWO = (0, 2)

# ── registry of every exported constant for the DSL engine ──────────────

CONSTANT_REGISTRY = {
    "ZERO": ZERO, "ONE": ONE, "TWO": TWO, "THREE": THREE,
    "FOUR": FOUR, "FIVE": FIVE, "SIX": SIX, "SEVEN": SEVEN,
    "EIGHT": EIGHT, "NINE": NINE, "TEN": TEN,
    "NEG_ONE": NEG_ONE, "NEG_TWO": NEG_TWO,
    "T": T, "F": F,
    "ORIGIN": ORIGIN, "UNITY": UNITY, "NEG_UNITY": NEG_UNITY,
    "DOWN": DOWN, "UP": UP, "RIGHT": RIGHT, "LEFT": LEFT,
    "DOWN_LEFT": DOWN_LEFT, "UP_RIGHT": UP_RIGHT,
    "TWO_BY_TWO": TWO_BY_TWO, "THREE_BY_THREE": THREE_BY_THREE,
    "TWO_BY_ZERO": TWO_BY_ZERO, "ZERO_BY_TWO": ZERO_BY_TWO,
}
