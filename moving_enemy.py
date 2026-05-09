"""
Moving Enemy Module (stretch feature)
An enemy patrols a fixed route through the maze.
The search algorithm must avoid cells occupied by the enemy
at the time-step the agent would arrive there.
"""

import numpy as np
from collections import deque
from maze_environment import MazeEnvironment, WALL


class MovingEnemy:
    """Enemy that patrols between two waypoints."""

    def __init__(self, env: MazeEnvironment, waypoint_a: tuple,
                 waypoint_b: tuple, speed: int = 1):
        self.env = env
        self.route = self._build_route(waypoint_a, waypoint_b)
        self.speed = speed
        self.t = 0

    def _build_route(self, a, b):
        """BFS shortest path between waypoints (the patrol route)."""
        q = deque([(a, [a])])
        seen = {a}
        while q:
            cur, path = q.popleft()
            if cur == b:
                return path
            for nb in self.env.get_neighbors(cur):
                if nb not in seen:
                    seen.add(nb)
                    q.append((nb, path + [nb]))
        return [a]  # fallback: stay in place

    def position_at(self, time_step: int) -> tuple:
        """Return enemy position at a given time step (ping-pong)."""
        cycle = (len(self.route) - 1) * 2
        if cycle == 0:
            return self.route[0]
        idx = time_step % cycle
        if idx < len(self.route):
            return self.route[idx]
        return self.route[cycle - idx]

    def occupied_cells(self, max_steps: int = 200):
        """Pre-compute occupied cell at each time step."""
        return {t: self.position_at(t) for t in range(max_steps)}


def a_star_with_enemy(env: MazeEnvironment, enemy: MovingEnemy):
    """
    A* that avoids the moving enemy.
    State = (row, col, time_step % cycle).
    """
    import heapq, time as _time
    from search_algorithms import SearchResult, _reconstruct

    t0 = _time.perf_counter()
    cycle = max((len(enemy.route) - 1) * 2, 1)

    counter  = 0
    open_set = [(env.heuristic(env.start), counter, env.start, 0)]
    came_from = {}
    g_score   = {(env.start, 0): 0}
    visited   = set()
    expanded  = 0

    while open_set:
        _, _, current, t = heapq.heappop(open_set)
        state = (current, t % cycle)
        if state in visited:
            continue
        visited.add(state)
        expanded += 1

        if current == env.goal:
            path = _reconstruct_path(came_from, state, env.start, cycle)
            elapsed = _time.perf_counter() - t0
            
            traps = 0
            for p in path:
                if env.grid[p[0]][p[1]] == 2:
                    traps += 1
                    
            cost = 0
            for p in path:
                cost += env.step_cost(p)
                
            return SearchResult("A*+Enemy", path, len(path), expanded,
                                elapsed, cost, True, traps)

        nt = t + 1
        enemy_pos = enemy.position_at(nt)
        for nb in env.get_neighbors(current):
            if nb == enemy_pos:
                continue                              # avoid enemy
            nstate = (nb, nt % cycle)
            ng = g_score[state] + env.step_cost(nb)
            if nstate not in g_score or ng < g_score[nstate]:
                g_score[nstate]  = ng
                came_from[nstate] = state
                f = ng + env.heuristic(nb)
                counter += 1
                heapq.heappush(open_set, (f, counter, nb, nt))

    elapsed = _time.perf_counter() - t0
    return SearchResult("A*+Enemy", [], 0, expanded, elapsed, 0, False)


def _reconstruct_path(came_from, state, start, cycle):
    path = []
    while state in came_from:
        path.append(state[0])
        state = came_from[state]
    path.append(start)
    path.reverse()
    return path