"""
Visualization Module
Matplotlib-based rendering of mazes, paths, risk maps, and comparison charts.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


# Custom colour map for the maze grid
MAZE_CMAP = ListedColormap([
    '#E8E8E8',   # 0  FREE   – light grey
    '#2C3E50',   # 1  WALL   – dark blue-grey
    '#E74C3C',   # 2  TRAP   – red
    '#E8E8E8',   # 3  (unused)
    '#E8E8E8',   # 4  (unused)
    '#27AE60',   # 5  START  – green
    '#F39C12',   # 6  GOAL   – gold
])

PATH_COLORS = {
    'BFS':             '#3498DB',
    'DFS':             '#9B59B6',
    'A*':              '#2ECC71',
    'Risk-Aware A*':   '#E67E22',
    'A*+Enemy':        '#1ABC9C',
}


# ── Single Maze Figure ────────────────────────────────────────
def visualize_maze(env, results_dict, risk_map=None, maze_name="Maze"):
    """
    Draw a 2×2 or 1×3 figure:
      Row 1 → BFS, DFS, A*
      Row 2 → Risk heatmap, Risk-Aware A*   (if risk_map provided)
    """
    has_risk = risk_map is not None
    nrows = 2 if has_risk else 1
    ncols = 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    fig.suptitle(f"Maze Solver Challenge — {maze_name}",
                 fontsize=15, fontweight='bold', y=1.01)

    if nrows == 1:
        axes = [axes]

    # ── Draw one panel ────────────────────────────────────────
    def _draw(ax, title, path=None, color=None):
        display = env.grid.astype(float).copy()
        if path:
            for r, c in path:
                if (r, c) != env.start and (r, c) != env.goal:
                    display[r][c] = 0.4        # tint path cells

        ax.imshow(display, cmap=MAZE_CMAP, vmin=0, vmax=6)

        if path and len(path) > 1:
            pts = np.array(path)
            ax.plot(pts[:, 1], pts[:, 0], '-', color=color,
                    linewidth=3.5, alpha=0.85, zorder=5)
            ax.plot(pts[:, 1], pts[:, 0], 'o', color=color,
                    markersize=4, alpha=0.6, zorder=6)

        ax.plot(env.start[1], env.start[0], 's', color='lime',
                markersize=14, zorder=10, markeredgecolor='black', markeredgewidth=1.5)
        ax.plot(env.goal[1], env.goal[0], '*', color='gold',
                markersize=18, zorder=10, markeredgecolor='black', markeredgewidth=1)

        ax.set_xticks(np.arange(-0.5, env.cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, env.rows, 1), minor=True)
        ax.grid(which='minor', color='grey', linewidth=0.3, alpha=0.4)
        ax.tick_params(which='both', bottom=False, left=False,
                       labelbottom=False, labelleft=False)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=8)

        legend_items = [
            mpatches.Patch(facecolor='#E8E8E8', edgecolor='grey', label='Free'),
            mpatches.Patch(facecolor='#2C3E50', label='Wall'),
            mpatches.Patch(facecolor='#E74C3C', label='Trap'),
            mpatches.Patch(facecolor='lime',    label='Start'),
            mpatches.Patch(facecolor='gold',    label='Goal'),
        ]
        if path:
            legend_items.append(mpatches.Patch(facecolor=color, alpha=0.7, label='Path'))
        ax.legend(handles=legend_items, loc='upper right', fontsize=6,
                  framealpha=0.85, edgecolor='grey')

    # Row 1: three core algorithms
    alg_list = ['BFS', 'DFS', 'A*']
    for i in range(len(alg_list)):
        alg = alg_list[i]
        res = results_dict.get(alg)
        path = res.path if res and res.found else None
        info = ""
        if res and res.found:
            info = (f"Len={res.path_length}  Cost={res.total_cost:.0f}  "
                    f"Exp={res.nodes_expanded}  Traps={res.trap_count}")
        _draw(axes[0][i], f"{alg}\n{info}", path, PATH_COLORS.get(alg))

    # Row 2: risk map + risk-aware A*
    if has_risk:
        im = axes[1][0].imshow(risk_map, cmap='YlOrRd', vmin=0, vmax=1)
        axes[1][0].plot(env.start[1], env.start[0], 's', color='lime',
                        markersize=14, zorder=10, markeredgecolor='black', markeredgewidth=1.5)
        axes[1][0].plot(env.goal[1], env.goal[0], '*', color='gold',
                        markersize=18, zorder=10, markeredgecolor='black', markeredgewidth=1)
        axes[1][0].set_title("NN Risk Prediction Map", fontsize=10, fontweight='bold')
        axes[1][0].set_xticks([]); axes[1][0].set_yticks([])
        fig.colorbar(im, ax=axes[1][0], fraction=0.046, pad=0.04,
                     label='P(trap)')

        ra = results_dict.get('Risk-Aware A*')
        if ra and ra.found:
            info = (f"Len={ra.path_length}  Cost={ra.total_cost:.0f}  "
                    f"Exp={ra.nodes_expanded}  Traps={ra.trap_count}")
            _draw(axes[1][1], f"Risk-Aware A*\n{info}", ra.path,
                PATH_COLORS['Risk-Aware A*'])
        else:
            axes[1][1].text(0.5, 0.5, 'No path found', ha='center',
                            transform=axes[1][1].transAxes, fontsize=12)
            axes[1][1].set_title("Risk-Aware A*", fontsize=10, fontweight='bold')

        # hide unused 4th cell if only 3 panels in row 2
        axes[1][2].axis('off')
        summary_text = _build_summary_text(results_dict)
        axes[1][2].text(0.05, 0.95, summary_text, transform=axes[1][2].transAxes,
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1][2].set_title("Summary", fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()


def _build_summary_text(results):
    header = f"{'Alg':<16} {'Len':>4} {'Cost':>6} {'Exp':>5} {'Trap':>4}\n"
    sep    = "-" * 40 + "\n"
    lines  = header + sep
    for name, r in results.items():
        if r.found:
            lines += f"{name:<16} {r.path_length:>4} {r.total_cost:>6.0f} {r.nodes_expanded:>5} {r.trap_count:>4}\n"
        else:
            lines += f"{name:<16}  FAILED\n"
    return lines


# ── Cross-Maze Comparison Bar Charts ──────────────────────────
def plot_comparison(all_results, maze_names):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Algorithm Comparison Across All Mazes",
                 fontsize=15, fontweight='bold')

    algorithms = ['BFS', 'DFS', 'A*', 'Risk-Aware A*']
    
    colors = []
    for a in algorithms:
        colors.append(PATH_COLORS[a])
        
    x = np.arange(len(maze_names))
    width = 0.2

    metrics = [
        ('Path Length',     'path_length'),
        ('Total Cost',      'total_cost'),
        ('Nodes Expanded',  'nodes_expanded'),
        ('Execution Time (ms)', 'time_taken'),
    ]

    for i in range(len(metrics)):
        title = metrics[i][0]
        key = metrics[i][1]
        
        ax = axes[i // 2][i % 2]
        
        for j in range(len(algorithms)):
            alg = algorithms[j]
            vals = []
            for mn in maze_names:
                r = all_results.get(mn, {}).get(alg)
                v = getattr(r, key, 0) if r and r.found else 0
                if key == 'time_taken':
                    v *= 1000
                vals.append(v)
            ax.bar(x + j * width, vals, width, label=alg,
                   color=colors[j], alpha=0.85, edgecolor='white')

        ax.set_ylabel(title)
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        
        labels = []
        for n in maze_names:
            labels.append(n.split(' - ')[0])
        ax.set_xticklabels(labels, fontsize=9)
        
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()