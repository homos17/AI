# Maze Solver Challenge
**Tagline:** Build a game agent that escapes faster and smarter.

**Course:** CET251 - Artificial Intelligence 

---

## Project Overview

A game agent navigates grid-based mazes filled with walls and dangerous traps to reach a defined goal. The core objective is to implement and compare different search strategies, evaluating their efficiency, cost, and optimality.

A Multi-Layer Perceptron (MLP) is trained on local maze features to predict the probability of nearby traps, creating a **Risk-Aware A\*** algorithm that actively avoids predicted dangers.

---

## Core AI Concepts

*   **Agents & Environments:** The agent interacts with the `MazeEnvironment` (perceiving valid neighbors, step costs, and traps) to make navigation decisions.
*   **BFS (Breadth-First Search):** Explores level-by-level; guarantees the shortest path in unweighted graphs (fewest steps).
*   **DFS (Depth-First Search):** Explores as deep as possible before backtracking; memory-efficient but not optimal.
*   **A\* Search:** Uses `f(n) = g(n) + h(n)` (Manhattan distance heuristic); guarantees the lowest-cost path while expanding fewer nodes.
*   **Neural Network Risk Prediction:** An MLP classifies cells as safe/traps based on local features, generating a probability map to guide the agent away from unseen dangers.

---

## Project Structure

```
maze_solver_challenge/
|
|-- main.py                # Entry point: training, solving, and visualization
|-- maze_environment.py    # Grid world, walls, traps, and valid moves
|-- search_algorithms.py   # BFS, DFS, A*, and Risk-Aware A*
|-- risk_predictor.py      # Neural network (Keras MLP) for trap prediction
|-- visualizer.py          # Matplotlib visualizations (paths, heatmaps, charts)
|-- pygame_visualizer.py   # Pygame animated visualizations (step-by-step paths)
|-- level_generator.py     # [Stretch] Procedural maze generation
|-- moving_enemy.py        # [Stretch] Moving obstacle and A* avoidance
|-- requirements.txt       # Python dependencies
```

---

## Setup & Installation

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

## How to Run

```bash
python main.py
```

### What you will see:
1. **NN Training Phase:** Console output showing MLP training accuracy and classification report.
2. **Console Grids:** ASCII representation of each maze with the paths found.
3. **Comparison Tables:** Side-by-side metrics (Path Length, Cost, Nodes Expanded, Time, Traps Hit) for all algorithms.
4. **Pygame Animations:** Step-by-step animated path for each algorithm (BFS, DFS, A*, Risk-Aware A*), plus NN risk heatmap and moving enemy demo.
5. **Matplotlib Windows:** Static visualizations with maze paths and bar charts comparing metrics across mazes.
6. **Stretch Features:** Procedurally generated random maze and moving enemy avoidance.

---

## Requirements Fulfilled

### Minimum Scope
*   [x] **3 Different Mazes**: Simple (10x10), Medium (12x12), Hard (15x15).
*   [x] **3 Search Algorithms**: BFS, DFS, A*.
*   [x] **1 Risk-Prediction Feature**: Neural network (Keras MLP) predicts trap probability from 13 local features.

### Stretch Ideas
*   [x] **Level Generator**: `level_generator.py` uses recursive backtracking to create random solvable mazes with multiple paths and scattered traps.
*   [x] **Moving Enemy**: `moving_enemy.py` features an enemy patrolling between waypoints; uses Time-Space A* to avoid collisions.

---

## Algorithm Comparison & Analysis

| Algorithm | Optimality | Trap Handling | Memory Usage | Speed (Nodes Expanded) |
| :--- | :--- | :--- | :--- | :--- |
| **BFS** | Optimal (fewest steps) | Ignores trap costs | High (Wide frontier) | Moderate |
| **DFS** | **Not optimal** | Ignores trap costs | Low (Deep stack) | Fast/Moderate (Luck) |
| **A\*** | Optimal (lowest cost) | Penalizes traps in `g(n)` | Moderate | Fast (Heuristic guided) |
| **Risk-Aware A\*** | Near-optimal | Avoids predicted traps | Moderate | Moderate (Inference overhead) |

### Key Findings
1. **A\* is the best general-purpose solver:** Uses Manhattan distance heuristic and trap penalties to find the cheapest route while expanding fewer nodes than BFS.
2. **BFS finds the shortest path, but not the safest:** Guarantees fewest steps but walks through traps blindly.
3. **DFS is unreliable:** Sometimes fast but frequently produces sub-optimal paths that hit multiple traps.
4. **Neural Networks add foresight:** Risk-Aware A* uses the MLP probability map to steer the agent away from corridors that look like traps.

---

## Tools Used

*   **Python 3.x**
*   **NumPy**: Grid manipulation and mathematical operations.
*   **Matplotlib**: Static maze visualizations, risk heatmaps, and comparison bar charts.
*   **Pygame**: Animated step-by-step path visualization and moving enemy demo.
*   **TensorFlow / Keras**: `Sequential` MLP for neural network risk prediction.
*   **Scikit-Learn**: Train/test split and classification metrics.