import numpy as np
from collections import deque

# Cell Type Constants
FREE  = 0
WALL  = 1
TRAP  = 2
START = 5
GOAL  = 6

SYMBOLS = {FREE: '.', WALL: '#', TRAP: 'X', START: 'S', GOAL: 'G'}

# Three Required Mazes
MAZES = {
    "Maze 1 - Simple (10x10)": {
        "grid": [
            [5, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 2, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 1, 1],
            [1, 1, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 1],
            [0, 1, 0, 0, 0, 2, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 6],
        ],
        "start": (0, 0),
        "goal":  (9, 9),
    },

    "Maze 2 - Medium (12x12)": {
        "grid": [
            [5, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [1, 1, 1, 0, 1, 2, 1, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],
            [0, 1, 2, 1, 1, 1, 0, 0, 0, 2, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 1, 1, 2, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 0, 1, 2, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 6],
        ],
        "start": (0, 0),
        "goal":  (11, 11),
    },

    "Maze 3 - Hard (15x15)": {
        "grid": [
            [5, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 2, 1, 1, 0, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 2, 1, 0, 0, 1, 1, 2, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 6],
        ],
        "start": (0, 0),
        "goal":  (14, 14),
    },
}


class MazeEnvironment:
    """
    The game world an agent interacts with.
    Provides: validity checks, neighbor generation, cost model, heuristic.
    """

    TRAP_PENALTY = 5          # extra cost for stepping on a trap
    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # 4-connected

    def __init__(self, maze_data: dict):
        self.grid  = np.array(maze_data["grid"])
        self.rows, self.cols = self.grid.shape
        self.start = tuple(maze_data["start"])
        self.goal  = tuple(maze_data["goal"])

    # Queries 
    def is_valid(self, pos: tuple) -> bool:
        """Within bounds and not a wall."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] != WALL

    def get_neighbors(self, pos: tuple) -> list:
        """Return list of valid 4-connected neighbours."""
        neighbors = []
        for dr, dc in self.DIRECTIONS:
            nr = pos[0] + dr
            nc = pos[1] + dc
            if self.is_valid((nr, nc)):
                neighbors.append((nr, nc))
        return neighbors

    def step_cost(self, pos: tuple) -> int:
        """Cost of entering a cell (traps are expensive)."""
        if self.grid[pos[0]][pos[1]] == TRAP:
            return 1 + self.TRAP_PENALTY
        return 1

    def heuristic(self, pos: tuple) -> int:
        """Manhattan distance to goal (admissible for 4-connected grids)."""
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])

    def trap_positions(self) -> list:
        """Return all trap coordinates."""
        positions = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == TRAP:
                    positions.append((r, c))
        return positions

    # Console Renderer
    def display_console(self, path=None, title="Maze"):
        """Print the maze (and optional path) to stdout."""
        canvas = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r, c) == self.start:
                    row.append('S')
                elif (r, c) == self.goal:
                    row.append('G')
                else:
                    row.append(SYMBOLS[self.grid[r][c]])
            canvas.append(row)

        if path:
            for r, c in path:
                if (r, c) != self.start and (r, c) != self.goal:
                    canvas[r][c] = 'o'

        print(f"\n  [{title}]")
        print("  " + "-" * (self.cols * 2 + 1))
        for row in canvas:
            print("  |" + " ".join(row) + "|")
        print("  " + "-" * (self.cols * 2 + 1))

        if path:
            print(f"   Legend: S=Start  G=Goal  #=Wall  X=Trap  o=Path")
        else:
            print(f"   Legend: S=Start  G=Goal  #=Wall  X=Trap  .=Free")