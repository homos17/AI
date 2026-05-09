#!/usr/bin/env python3
"""
MAZE SOLVER CHALLENGE
Build a game agent that escapes faster and smarter.
Compare search algorithms in a game-like environment.

Core AI Concepts:
  • Agents & Environments
  • BFS / DFS / A* Search
  • Neural-network risk prediction

Minimum Scope (all covered):
  ✅ 3 different mazes
  ✅ 3 search algorithms (BFS, DFS, A*)
  ✅ 1 risk-prediction feature (MLP)

Stretch Features:
  ✅ Level generator
  ✅ Moving enemy
"""

import sys
import time
import numpy as np

from maze_environment import MazeEnvironment, MAZES, FREE, WALL, TRAP
from search_algorithms import (
    breadth_first_search,
    depth_first_search,
    a_star_search,
    risk_aware_a_star,
    SearchResult,
)
from risk_predictor import RiskPredictor
from level_generator import generate_maze
from visualizer import visualize_maze, plot_comparison

# ═══════════════════════════════════════════════════════════════
#  UTILITY PRINTING
# ═══════════════════════════════════════════════════════════════

SEP = "=" * 70

def heading(text):
    print(f"\n{SEP}")
    print(f"  {text}")
    print(SEP)


def print_result_table(results):
    hdr = (f"  {'Algorithm':<18}{'Found':<9}{'Path':>6}{'Cost':>8}"
           f"{'Expanded':>11}{'Time ms':>11}{'Traps':>7}")
    print(f"\n{hdr}")
    print("  " + "-" * 66)
    for name, r in results.items():
        ok = "Yes" if r.found else "No"
        t_ms = r.time_taken * 1000
        print(f"  {name:<18}{ok:<9}{r.path_length:>6}{r.total_cost:>8.1f}"
              f"{r.nodes_expanded:>11}{t_ms:>11.4f}{r.trap_count:>7}")
    print()


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    heading("MAZE SOLVER CHALLENGE")
    print("""
  A game agent must navigate mazes with walls, traps, and
  changing conditions.  We compare BFS, DFS, A*, and a
  Neural-Network Risk-Aware A* across three mazes.

  Stretch features included:
    • Procedural level generator
    • Moving enemy avoidance
    """)

    # ── 1. Train Neural-Network Risk Predictor ─────────────────
    predictor = RiskPredictor()
    predictor.train(MAZES)

    all_results = {}
    maze_names  = list(MAZES.keys())

    # ── 2. Solve Each Maze ────────────────────────────────────
    for maze_name, maze_data in MAZES.items():
        heading(f"SOLVING: {maze_name}")
        env = MazeEnvironment(maze_data)
        n_traps = 0
        for row in maze_data["grid"]:
            n_traps += row.count(TRAP)
        print(f"  Size  : {env.rows}×{env.cols}")
        print(f"  Start : {env.start}   Goal : {env.goal}")
        print(f"  Traps : {n_traps}")

        env.display_console(title=maze_name)

        # Run three core algorithms
        results = {}
        for label, func in [("BFS", breadth_first_search),
                            ("DFS", depth_first_search),
                            ("A*", a_star_search)]:
            print(f"  Running {label}...", end=" ", flush=True)
            results[label] = func(env)
            print("done" if results[label].found else "FAILED")

        # Neural-network risk map → Risk-Aware A*
        print("  Generating NN risk map...", end=" ", flush=True)
        risk_map = predictor.predict_risk_map(env)
        print("done")
        print("  Running Risk-Aware A*...", end=" ", flush=True)
        results["Risk-Aware A*"] = risk_aware_a_star(env, risk_map)
        print("done" if results["Risk-Aware A*"].found else "FAILED")

        # Display comparison table
        print_result_table(results)

        # Console paths
        if results["A*"].found:
            env.display_console(results["A*"].path, title="A* Path")
        if results["Risk-Aware A*"].found:
            env.display_console(results["Risk-Aware A*"].path,
                                title="Risk-Aware A* Path")

        all_results[maze_name] = results

        # Matplotlib visualisation
        visualize_maze(env, results, risk_map, maze_name)

    # ── 3. Cross-Maze Comparison Charts ───────────────────────
    heading("CROSS-MAZE COMPARISON")
    plot_comparison(all_results, maze_names)

    # ── 4. Stretch: Procedural Maze ───────────────────────────
    heading("STRETCH — Procedural Level Generator")
    gen_data = generate_maze(17, 17, trap_density=0.10, extra_openings=0.20)
    gen_env  = MazeEnvironment(gen_data)
    gen_env.display_console(title="Generated Maze")

    gen_results = {}
    for label, func in [("BFS", breadth_first_search),
                        ("DFS", depth_first_search),
                        ("A*", a_star_search)]:
        gen_results[label] = func(gen_env)

    gen_risk = predictor.predict_risk_map(gen_env)
    gen_results["Risk-Aware A*"] = risk_aware_a_star(gen_env, gen_risk)
    print_result_table(gen_results)
    visualize_maze(gen_env, gen_results, gen_risk, "Generated Maze")

    # ── 5. Stretch: Moving Enemy ──────────────────────────────
    heading("STRETCH — Moving Enemy Avoidance")
    from moving_enemy import MovingEnemy, a_star_with_enemy

    # Pick first maze for the demo
    demo_data = MAZES[maze_names[0]]
    demo_env  = MazeEnvironment(demo_data)

    # Find two waypoints for enemy patrol
    free_cells = []
    for r in range(demo_env.rows):
        for c in range(demo_env.cols):
            if demo_env.is_valid((r, c)):
                free_cells.append((r, c))
                
    wp_a = free_cells[len(free_cells) // 4]
    wp_b = free_cells[3 * len(free_cells) // 4]

    enemy = MovingEnemy(demo_env, wp_a, wp_b, speed=1)
    print(f"  Enemy patrols {wp_a} <-> {wp_b}")
    print(f"  Patrol route length: {len(enemy.route)}")

    enemy_result = a_star_with_enemy(demo_env, enemy)
    print(f"  A*+Enemy result: {enemy_result}")
    if enemy_result.found:
        demo_env.display_console(enemy_result.path, title="A* with Moving Enemy")

    # ── 6. Final Analysis ─────────────────────────────────────
    heading("ALGORITHM ANALYSIS & CONCLUSIONS")
    print("""
  +-------------------+----------------------------------------------------+
  | Algorithm         | Characteristics                                    |
  +-------------------+----------------------------------------------------+
  | BFS               | + Shortest path (fewest steps)                     |
  |                   | + Complete & optimal for unweighted graphs         |
  |                   | - Ignores trap costs                               |
  |                   | - High memory (wide frontier)                      |
  +-------------------+----------------------------------------------------+
  | DFS               | + Low memory (deep stack)                           |
  |                   | + Fast on some mazes (luck-dependent)               |
  |                   | - NOT optimal (may find very long paths)            |
  |                   | - NOT complete in infinite spaces                   |
  +-------------------+----------------------------------------------------+
  | A*                | + Optimal with admissible heuristic                 |
  |                   | + Considers trap costs in g(n)                      |
  |                   | + Fewer expansions than BFS/DFS                    |
  |                   | - More complex; needs good heuristic               |
  +-------------------+----------------------------------------------------+
  | Risk-Aware A*     | + Uses NN-predicted trap probabilities              |
  |                   | + Avoids predicted risky cells                      |
  |                   | + Often finds paths with FEWER traps hit            |
  |                   | - Depends on training data quality                  |
  |                   | - Slight overhead for risk inference                |
  +-------------------+----------------------------------------------------+

  KEY FINDINGS:
  1. A* consistently finds the lowest-COST path (accounting for traps).
  2. BFS finds the fewest-STEPS path but ignores trap penalties.
  3. DFS is unpredictable -- sometimes fast, often suboptimal.
  4. Risk-Aware A* with NN predictions can avoid traps the agent
     hasn't "seen" yet, reducing trap encounters significantly.
  5. The neural network achieves good trap prediction accuracy
     from purely LOCAL features (neighbourhood topology, distances).
""")


if __name__ == "__main__":
    main()