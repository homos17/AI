"""
Procedural Maze Generator (stretch feature)
Uses recursive back-tracking to create random solvable mazes,
then scatters traps based on a density parameter.
"""

import numpy as np
import random
from maze_environment import MazeEnvironment, FREE, WALL, TRAP, START, GOAL


def generate_maze(rows: int, cols: int, trap_density: float = 0.05,
                  seed: int = None) -> dict:
    """
    Create a random maze with guaranteed start→goal path.
    1. Fill grid with walls.
    2. Carve passages with recursive back-tracking (DFS).
    3. Place traps on random free cells.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Ensure odd dimensions for clean corridors
    rows = rows if rows % 2 == 1 else rows + 1
    cols = cols if cols % 2 == 1 else cols + 1

    grid = np.full((rows, cols), WALL, dtype=int)

    # Carve with iterative DFS (stack-based to avoid recursion limit)
    start_cell = (1, 1)
    stack = [start_cell]
    grid[start_cell] = FREE
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]

    while stack:
        r, c = stack[-1]
        neighbours = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 1 <= nr < rows - 1 and 1 <= nc < cols - 1 and grid[nr][nc] == WALL:
                neighbours.append((nr, nc, r + dr // 2, c + dc // 2))
        if neighbours:
            nr, nc, wr, wc = random.choice(neighbours)
            grid[wr][wc] = FREE
            grid[nr][nc] = FREE
            stack.append((nr, nc))
        else:
            stack.pop()

    # Choose start (top-left) and goal (bottom-right)
    free_cells = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == FREE:
                free_cells.append((r, c))
                
    start = (1, 1)
    goal  = (rows - 2, cols - 2)
    grid[start] = START
    grid[goal]  = GOAL

    # Scatter traps
    trap_candidates = []
    for r, c in free_cells:
        if (r, c) != start and (r, c) != goal:
            trap_candidates.append((r, c))
            
    n_traps = max(1, int(len(trap_candidates) * trap_density))
    for r, c in random.sample(trap_candidates, min(n_traps, len(trap_candidates))):
        grid[r][c] = TRAP

    return {"grid": grid.tolist(), "start": start, "goal": goal}