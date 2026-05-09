"""
Neural-Network Risk Predictor
Learns trap probability from local maze features.
Core AI Concept: Neural-network add-on for agent reasoning.
"""

import numpy as np
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

from maze_environment import MazeEnvironment

from tensorflow import keras
from tensorflow.keras import layers


class RiskPredictor:
    """
    Keras Neural Network that predicts P(trap | local_features).
    Matches the Sequential model concepts learned during the semester.
    """

    def __init__(self):
        self.model = None
        self.trained = False

    def build_model(self, num_features):
        """Builds a Keras Sequential model like in breast_cancer.py"""
        model = keras.Sequential()
        model.add(layers.Input(shape=(num_features,)))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    # ── Feature Engineering ────────────────────────────────────
    @staticmethod
    def _dist_to_nearest(env, pos, cell_type, max_depth=6):
        """BFS distance to nearest cell of `cell_type` (capped)."""
        q = deque([(pos, 0)])
        seen = {pos}
        while q:
            cur, d = q.popleft()
            if d > 0 and env.grid[cur[0]][cur[1]] == cell_type:
                return d
            if d >= max_depth:
                break
            for nb in env.get_neighbors(cur):
                if nb not in seen:
                    seen.add(nb)
                    q.append((nb, d + 1))
        return max_depth + 1

    def extract_features(self, env, pos):
        """
        13-dimensional feature vector for a single cell:
          [wall_3x3, trap_3x3, free_3x3,
           wall_dir4, trap_dir4,
           dist_goal, dist_start, dist_nearest_trap,
           norm_r, norm_c,
           is_dead_end, is_corridor, is_junction]
        """
        r, c = pos
        rows, cols = env.rows, env.cols
        grid = env.grid

        # 3×3 neighbourhood counts
        w3 = t3 = f3 = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    v = grid[nr][nc]
                    if v == 1: w3 += 1
                    elif v == 2: t3 += 1
                    elif v == 0: f3 += 1

        # 4-directional counts
        wd = td = 0
        free_nb = 0
        for dr, dc in env.DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr][nc] == 1: wd += 1
                elif grid[nr][nc] == 2: td += 1
                if env.is_valid((nr, nc)):
                    free_nb += 1
            else:
                wd += 1                          # out-of-bounds ≡ wall

        return [
            w3, t3, f3, wd, td,
            abs(r - env.goal[0]) + abs(c - env.goal[1]),   # dist to goal
            abs(r - env.start[0]) + abs(c - env.start[1]), # dist to start
            self._dist_to_nearest(env, pos, 2),            # dist to trap
            r / rows, c / cols,                            # normalised pos
            1 if free_nb <= 1 else 0,                      # dead-end
            1 if free_nb == 2 else 0,                      # corridor
            1 if free_nb >= 3 else 0,                      # junction
        ]

    # ── Data Generation ────────────────────────────────────────
    def generate_training_data(self, mazes_dict):
        X, y = [], []
        for data in mazes_dict.values():
            env = MazeEnvironment(data)
            for r in range(env.rows):
                for c in range(env.cols):
                    if env.grid[r][c] != 1:                # skip walls
                        X.append(self.extract_features(env, (r, c)))
                        y.append(1 if env.grid[r][c] == 2 else 0)
        return np.array(X), np.array(y)

    # ── Training ───────────────────────────────────────────────
    def train(self, mazes_dict):
        X, y = self.generate_training_data(mazes_dict)

        trap_count = 0
        for val in y:
            if val == 1:
                trap_count += 1
        safe_count = len(y) - trap_count

        print("\n" + "=" * 62)
        print("  NEURAL NETWORK RISK PREDICTOR — Training Phase (Keras)")
        print("=" * 62)
        print(f"  Samples        : {len(X)}")
        print(f"  Trap cells     : {trap_count}")
        print(f"  Safe cells     : {safe_count}")
        print(f"  Features       : {X.shape[1]}")

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = self.build_model(X.shape[1])
        
        print("  Training model... (this may take a few seconds)")
        # Train model using fit
        self.model.fit(
            X_tr, y_tr,
            epochs=100,
            batch_size=16,
            verbose=1
        )
        self.trained = True

        acc_train = self.model.evaluate(X_tr, y_tr, verbose=0)[1]
        acc_test  = self.model.evaluate(X_te, y_te, verbose=0)[1]

        print(f"\n  Train accuracy : {acc_train:.4f}")
        print(f"  Test  accuracy : {acc_test:.4f}")
        print(f"\n  Classification report (test):")
        
        y_pred_probs = self.model.predict(X_te, verbose=0)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        
        report = classification_report(y_te, y_pred,
                                       target_names=["Safe", "Trap"], zero_division=0)
        for line in report.split('\n'):
            print(f"    {line}")
        print("=" * 62)
        return acc_test

    # ── Inference ──────────────────────────────────────────────
    def predict_risk_map(self, env):
        """Return a rows×cols array of P(trap) for every non-wall cell."""
        if not self.trained:
            raise RuntimeError("Model has not been trained yet.")
        risk = np.zeros((env.rows, env.cols))
        
        cells = []
        feats = []
        for r in range(env.rows):
            for c in range(env.cols):
                if env.grid[r][c] != 1:
                    cells.append((r, c))
                    feats.append(self.extract_features(env, (r, c)))
                    
        if len(feats) > 0:
            feats_array = np.array(feats)
            probs = self.model.predict(feats_array, verbose=0).flatten()
            for i in range(len(cells)):
                r = cells[i][0]
                c = cells[i][1]
                prob = probs[i]
                risk[r][c] = prob
                
        return risk