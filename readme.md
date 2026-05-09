# 🏃 Maze Solver Challenge
**Tagline:** Build a game agent that escapes faster and smarter.

**Course:** CET251 - Artificial Intelligence 

---

## 📖 Project Overview

In this project, a game agent must navigate grid-based mazes filled with walls and dangerous traps to reach a defined goal. The core objective is not just to find *a* path, but to implement and compare different search strategies, evaluating their efficiency, cost, and optimality. 

As a neural-network add-on, a Multi-Layer Perceptron (MLP) is trained on local maze features to predict the probability of nearby traps, creating a **Risk-Aware A\*** algorithm that actively avoids predicted dangers.

---

## 🧠 Core AI Concepts

*   **Agents & Environments:** The agent interacts with the `MazeEnvironment` (perceiving valid neighbors, step costs, and traps) to make navigation decisions.
*   **BFS (Breadth-First Search):** Explores level-by-level; guarantees the shortest path in unweighted graphs (fewest steps).
*   **DFS (Depth-First Search):** Explores as deep as possible before backtracking; memory-efficient but not optimal.
*   **A\* Search:** Uses `f(n) = g(n) + h(n)` (Manhattan distance heuristic); guarantees the lowest-cost path while expanding fewer nodes.
*   **Neural Network Risk Prediction:** An MLP classifies cells as safe/traps based on local features, generating a probability map to guide the agent away from unseen dangers.

---

## 📁 Project Structure

```
maze_solver_challenge/
│
├── main.py              # Entry point: orchestrates training, solving, and visualization
├── maze_environment.py  # Defines the grid world, walls, traps, and valid moves
├── search_algorithms.py # Implementations of BFS, DFS, A*, and Risk-Aware A*
├── risk_predictor.py    # Neural network (Keras MLP) for trap probability prediction
├── visualizer.py        # Matplotlib visualizations (paths, risk heatmaps, bar charts)
├── level_generator.py   # [Stretch] Procedural maze generation via recursive backtracking
├── moving_enemy.py      # [Stretch] Time-stepping moving obstacle and A* avoidance
└── requirements.txt     # Python dependencies
```

---

## ⚙️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-link>
   cd maze_solver_challenge
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

Execute the main script to train the neural network, run all algorithms across the 3 mazes, display console outputs, and generate the matplotlib visualizations:

```bash
python main.py
```

### What you will see:
1. **NN Training Phase:** Console output showing the MLP training accuracy and classification report.
2. **Console Grids:** ASCII representation of each maze, followed by the paths found by A* and Risk-Aware A*.
3. **Comparison Tables:** Side-by-side metrics (Path Length, Cost, Nodes Expanded, Time, Traps Hit) for BFS, DFS, A*, and Risk-Aware A*.
4. **Matplotlib Windows:** 
   - 2x3 grid showing maze paths for all 4 algorithms + the NN Risk Heatmap.
   - Bar charts comparing metrics across all mazes.
5. **Stretch Features:** Output from the procedurally generated maze and the moving enemy simulation.

---

## ✅ Requirements Fulfilled

### Minimum Scope
*   [x] **3 Different Mazes**: Simple (10x10), Medium (12x12), Hard (15x15).
*   [x] **3 Search Algorithms**: BFS, DFS, A*.
*   [x] **1 Risk-Prediction Feature**: Neural network (MLP) that predicts trap probability from 13 local features (wall/trap counts in 3x3 and 4-directional neighborhoods, distances, topology like dead-ends/corridors).

### Stretch Ideas
*   [x] **Level Generator**: `level_generator.py` uses recursive backtracking to create random, solvable mazes with scattered traps.
*   [x] **Moving Enemy**: `moving_enemy.py` features an enemy patrolling between waypoints; uses Time-Space A* (`State = position + time_step`) to avoid collisions.

---

## 📊 Algorithm Comparison & Analysis

The project heavily emphasizes algorithmic comparison over graphics. Below is the comparative analysis of the algorithms:

| Algorithm | Optimality | Trap Handling | Memory Usage | Speed (Nodes Expanded) |
| :--- | :--- | :--- | :--- | :--- |
| **BFS** | Optimal (fewest steps) | Ignores trap costs | High (Wide frontier) | Moderate |
| **DFS** | **Not optimal** | Ignores trap costs | Low (Deep stack) | Fast/Moderate (Luck) |
| **A\*** | Optimal (lowest cost) | Penalizes traps in `g(n)` | Moderate | Fast (Heuristic guided) |
| **Risk-Aware A\*** | Near-optimal | Avoids predicted traps | Moderate | Moderate (Inference overhead) |

### Key Findings
1. **A\* is the best general-purpose solver:** By using the Manhattan distance heuristic and factoring trap penalties into the step cost, A* consistently finds the cheapest route while expanding far fewer nodes than BFS.
2. **BFS finds the shortest path, but not the safest:** BFS guarantees the fewest steps, but blindly walks through traps because it ignores step costs.
3. **DFS is unreliable:** While sometimes fast due to its aggressive deep diving, DFS frequently produces sub-optimal, convoluted paths that hit multiple traps.
4. **Neural Networks add "foresight":** The Risk-Aware A* uses the MLP's probability map (`P(trap)`) to steer the agent away from corridors that *look* like traps, often finding paths with zero trap encounters even when A* might take a risky shortcut.

---

## 🛠️ Tools Used

*   **Python 3.x**
*   **NumPy**: Grid manipulation and mathematical operations.
*   **Matplotlib**: Visualizing maze grids, paths, risk heatmaps, and comparison bar charts.
*   **TensorFlow / Keras**: `Sequential` MLP for the neural network risk-prediction add-on.