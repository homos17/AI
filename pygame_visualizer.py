
import pygame
import sys
import time
import numpy as np

# Colours
BLACK     = (20, 20, 20)
WHITE     = (240, 240, 240)
GREY      = (80, 80, 80)
DARK_GREY = (45, 45, 50)

WALL_COL  = (44, 62, 80)
FREE_COL  = (220, 220, 215)
TRAP_COL  = (231, 76, 60)
START_COL = (39, 174, 96)
GOAL_COL  = (243, 156, 18)

AGENT_COL     = (52, 152, 219)
VISITED_COL   = (174, 214, 241)
ENEMY_COL     = (142, 68, 173)

PATH_COLORS = {
    'BFS':           (52, 152, 219),
    'DFS':           (155, 89, 182),
    'A*':            (46, 204, 113),
    'Risk-Aware A*': (230, 126, 34),
    'A*+Enemy':      (26, 188, 156),
}

FREE = 0
WALL = 1
TRAP = 2
START_VAL = 5
GOAL_VAL = 6

CELL_SIZE = 40
FPS = 60


def _get_cell_color(val):
    if val == WALL:
        return WALL_COL
    elif val == TRAP:
        return TRAP_COL
    elif val == START_VAL:
        return START_COL
    elif val == GOAL_VAL:
        return GOAL_COL
    else:
        return FREE_COL


def _draw_maze(screen, grid, rows, cols):
    for r in range(rows):
        for c in range(cols):
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, _get_cell_color(grid[r][c]), rect)
            pygame.draw.rect(screen, DARK_GREY, rect, 1)


def _draw_dot(screen, r, c, color, shrink=4):
    cx = c * CELL_SIZE + CELL_SIZE // 2
    cy = r * CELL_SIZE + CELL_SIZE // 2
    pygame.draw.circle(screen, color, (cx, cy), CELL_SIZE // 2 - shrink)


def _draw_text(screen, text, x, y, size=20, color=WHITE):
    font = pygame.font.SysFont("consolas", size, bold=True)
    surf = font.render(str(text), True, color)
    screen.blit(surf, (x, y))


def _handle_events():
    """Process events, return False if user wants to quit."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return False
    return True


def run_pygame_demo(env, results_dict, risk_map=None, maze_name="Maze",
                    enemy=None, enemy_result=None):
    """
    Full pygame demo - auto-plays each algorithm animation.
    """
    pygame.init()

    rows, cols = env.rows, env.cols
    width = max(cols * CELL_SIZE, 400)
    height = rows * CELL_SIZE + 90

    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    # Animate one algorithm
    def animate_single(path, alg_name, step_delay_ms=80):
        #Animate agent walking along path. Returns False if quit.
        color = PATH_COLORS.get(alg_name, AGENT_COL)
        trail = []
        last_step_time = pygame.time.get_ticks()

        idx = 0
        finished = False
        finish_time = 0

        while True:
            if not _handle_events():
                return False

            now = pygame.time.get_ticks()

            # advance one step
            if not finished and now - last_step_time >= step_delay_ms:
                last_step_time = now
                if idx < len(path):
                    trail.append(path[idx])
                    idx += 1
                else:
                    finished = True
                    finish_time = now

            # after finishing, wait 1.5 seconds then auto-continue
            if finished and now - finish_time >= 1500:
                return True

            # ── Draw ──
            screen.fill(BLACK)
            _draw_maze(screen, env.grid, rows, cols)

            # path trail
            for i in range(len(trail)):
                r, c = trail[i]
                if (r, c) != env.start and (r, c) != env.goal:
                    bright = 120 + int(135 * i / max(len(path), 1))
                    tcol = (min(color[0], bright), min(color[1], bright), min(color[2], bright))
                    _draw_dot(screen, r, c, tcol, shrink=6)

            # start + goal
            _draw_dot(screen, env.start[0], env.start[1], START_COL, shrink=3)
            _draw_dot(screen, env.goal[0], env.goal[1], GOAL_COL, shrink=3)

            # agent dot (pulsing)
            if trail:
                ar, ac = trail[-1]
                pulse = abs(int(6 * np.sin(now * 0.008)))
                _draw_dot(screen, ar, ac, color, shrink=2 + pulse)

            # info panel
            panel_y = rows * CELL_SIZE
            pygame.draw.rect(screen, (30, 30, 35), (0, panel_y, width, 90))
            pygame.draw.line(screen, GREY, (0, panel_y), (width, panel_y), 2)

            traps = sum(1 for r, c in trail if env.grid[r][c] == TRAP)
            cost = sum(env.step_cost((r, c)) for r, c in trail)

            _draw_text(screen, alg_name, 15, panel_y + 8, 24, color)
            _draw_text(screen, f"Step: {idx}/{len(path)}   Cost: {cost}   Traps hit: {traps}",
                       15, panel_y + 40, 16, WHITE)

            if finished:
                _draw_text(screen, "DONE!", width - 80, panel_y + 8, 20, START_COL)
                _draw_text(screen, maze_name, width - 300, panel_y + 65, 13, GREY)

            _draw_text(screen, "ESC = skip all", width - 140, panel_y + 65, 12, GREY)

            pygame.display.flip()
            clock.tick(FPS)

    # Animate enemy + agent simultaneously
    def animate_with_enemy(agent_path, enemy_obj, step_delay_ms=120):
        #Animate agent AND enemy moving together.
        color = PATH_COLORS.get('A*+Enemy', AGENT_COL)
        trail = []
        last_step_time = pygame.time.get_ticks()

        idx = 0
        finished = False
        finish_time = 0

        while True:
            if not _handle_events():
                return False

            now = pygame.time.get_ticks()

            if not finished and now - last_step_time >= step_delay_ms:
                last_step_time = now
                if idx < len(agent_path):
                    trail.append(agent_path[idx])
                    idx += 1
                else:
                    finished = True
                    finish_time = now

            if finished and now - finish_time >= 2000:
                return True

            # Draw
            screen.fill(BLACK)
            _draw_maze(screen, env.grid, rows, cols)

            # path trail
            for i in range(len(trail)):
                r, c = trail[i]
                if (r, c) != env.start and (r, c) != env.goal:
                    _draw_dot(screen, r, c, color, shrink=7)

            _draw_dot(screen, env.start[0], env.start[1], START_COL, shrink=3)
            _draw_dot(screen, env.goal[0], env.goal[1], GOAL_COL, shrink=3)

            # enemy (purple pulsing square)
            epos = enemy_obj.position_at(idx)
            er, ec = epos
            pulse = abs(int(4 * np.sin(now * 0.01)))
            erect = pygame.Rect(ec * CELL_SIZE + 5 + pulse,
                                er * CELL_SIZE + 5 + pulse,
                                CELL_SIZE - 10 - 2 * pulse,
                                CELL_SIZE - 10 - 2 * pulse)
            pygame.draw.rect(screen, ENEMY_COL, erect, border_radius=5)

            # agent dot
            if trail:
                ar, ac = trail[-1]
                apulse = abs(int(5 * np.sin(now * 0.008)))
                _draw_dot(screen, ar, ac, AGENT_COL, shrink=2 + apulse)

            # info panel
            panel_y = rows * CELL_SIZE
            pygame.draw.rect(screen, (30, 30, 35), (0, panel_y, width, 90))
            pygame.draw.line(screen, GREY, (0, panel_y), (width, panel_y), 2)

            _draw_text(screen, "A* + Moving Enemy", 15, panel_y + 8, 24, AGENT_COL)
            _draw_text(screen, f"Step: {idx}/{len(agent_path)}   Enemy at: {epos}",
                       15, panel_y + 40, 16, WHITE)

            if finished:
                _draw_text(screen, "DONE!", width - 80, panel_y + 8, 20, START_COL)

            _draw_text(screen, "Purple = Enemy   Blue = Agent", 15, panel_y + 65, 13, GREY)

            pygame.display.flip()
            clock.tick(FPS)

    # Show risk heatmap
    def show_heatmap(risk):
        # Display NN risk heatmap for 3 seconds.
        start_time = pygame.time.get_ticks()

        while True:
            if not _handle_events():
                return False

            now = pygame.time.get_ticks()
            if now - start_time >= 3000:
                return True

            screen.fill(BLACK)

            for r in range(rows):
                for c in range(cols):
                    rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    if env.grid[r][c] == WALL:
                        pygame.draw.rect(screen, WALL_COL, rect)
                    else:
                        rv = risk[r][c]
                        red = int(min(255, rv * 500))
                        green = int(max(0, 200 - rv * 400))
                        pygame.draw.rect(screen, (red, green, 50), rect)
                    pygame.draw.rect(screen, DARK_GREY, rect, 1)

            # mark actual traps with white dot
            for r in range(rows):
                for c in range(cols):
                    if env.grid[r][c] == TRAP:
                        cx = c * CELL_SIZE + CELL_SIZE // 2
                        cy = r * CELL_SIZE + CELL_SIZE // 2
                        pygame.draw.circle(screen, WHITE, (cx, cy), 5)

            _draw_dot(screen, env.start[0], env.start[1], START_COL, shrink=3)
            _draw_dot(screen, env.goal[0], env.goal[1], GOAL_COL, shrink=3)

            panel_y = rows * CELL_SIZE
            pygame.draw.rect(screen, (30, 30, 35), (0, panel_y, width, 90))
            pygame.draw.line(screen, GREY, (0, panel_y), (width, panel_y), 2)
            _draw_text(screen, "Neural Network Risk Heatmap", 15, panel_y + 8, 22, GOAL_COL)
            _draw_text(screen, "Green = Safe   Red = Danger   White dots = Actual traps",
                       15, panel_y + 40, 14, GREY)

            pygame.display.flip()
            clock.tick(30)

    # RUN THE FULL DEMO
    pygame.display.set_caption(f"Maze Solver - {maze_name}")

    # animate each algorithm in order
    for alg in ['BFS', 'DFS', 'A*', 'Risk-Aware A*']:
        res = results_dict.get(alg)
        if res and res.found and res.path:
            ok = animate_single(res.path, alg, step_delay_ms=80)
            if not ok:
                pygame.quit()
                return

    # show risk heatmap
    if risk_map is not None:
        ok = show_heatmap(risk_map)
        if not ok:
            pygame.quit()
            return

    # animate enemy demo
    if enemy is not None and enemy_result is not None and enemy_result.found:
        ok = animate_with_enemy(enemy_result.path, enemy, step_delay_ms=120)
        if not ok:
            pygame.quit()
            return

    pygame.quit()
