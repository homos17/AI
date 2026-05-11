import time

class SearchResult:
    def __init__(self, algorithm, path, path_length, nodes_expanded,
                 time_taken, total_cost, found, trap_count=0):
        self.algorithm      = algorithm
        self.path           = path
        self.path_length    = path_length
        self.nodes_expanded = nodes_expanded
        self.time_taken     = time_taken
        self.total_cost     = total_cost
        self.found          = found 
        self.trap_count     = trap_count

    def __repr__(self):
        status = "SUCCESS" if self.found else "FAILED"
        return (f"<{self.algorithm} {status} path={self.path_length} "
                f"cost={self.total_cost:.1f} expanded={self.nodes_expanded} "
                f"traps={self.trap_count}>")


def breadth_first_search(env):
    t0 = time.perf_counter()
    visited = set()
    queue = [[env.start]]
    expanded = 0

    while queue:
        path = queue.pop(0)
        node = path[-1]

        if node in visited:
            continue
        visited.add(node)
        expanded += 1

        if node == env.goal:
            elapsed = time.perf_counter() - t0
            cost = 0
            for p in path:
                cost += env.step_cost(p)
                
            traps = 0
            for p in path:
                r = p[0]
                c = p[1]
                if env.grid[r][c] == 2:
                    traps += 1
                    
            return SearchResult("BFS", path, len(path), expanded, elapsed, cost, True, traps)
        else:
            adjacent_nodes = env.get_neighbors(node)
            for node2 in adjacent_nodes:
                new_path = path.copy()
                new_path.append(node2)
                queue.append(new_path)

    elapsed = time.perf_counter() - t0
    return SearchResult("BFS", [], 0, expanded, elapsed, 0, False)


def depth_first_search(env):
    t0 = time.perf_counter()
    visited = set()
    stack = [[env.start]]
    expanded = 0

    while stack:
        path = stack.pop()
        node = path[-1]

        if node in visited:
            continue
        visited.add(node)
        expanded += 1

        if node == env.goal:
            elapsed = time.perf_counter() - t0
            cost = 0
            for p in path:
                cost += env.step_cost(p)
                
            traps = 0
            for p in path:
                r = p[0]
                c = p[1]
                if env.grid[r][c] == 2:
                    traps += 1
                    
            return SearchResult("DFS", path, len(path), expanded, elapsed, cost, True, traps)
        else:
            adjacent_nodes = env.get_neighbors(node)
            for node2 in adjacent_nodes:
                new_path = path.copy()
                new_path.append(node2)
                stack.append(new_path)

    elapsed = time.perf_counter() - t0
    return SearchResult("DFS", [], 0, expanded, elapsed, 0, False)


def a_star_search(env):
    t0 = time.perf_counter()
    
    def path_f_cost(path):
        g_cost = 0
        for (node, cost) in path:
            g_cost += cost
        last_node = path[-1][0]
        h_cost = env.heuristic(last_node)
        f_cost = g_cost + h_cost
        return f_cost, last_node
        
    visited = set()
    queue = [[(env.start, 0)]]
    expanded = 0
    
    while queue:
        queue.sort(key=path_f_cost)
        path = queue.pop(0)
        node = path[-1][0]
        
        if node in visited:
            continue
        visited.add(node)
        expanded += 1
        
        if node == env.goal:
            clean_path = []
            for p in path:
                clean_path.append(p[0])
                
            elapsed = time.perf_counter() - t0
            
            cost = 0
            for p in clean_path:
                cost += env.step_cost(p)
                
            traps = 0
            for p in clean_path:
                r = p[0]
                c = p[1]
                if env.grid[r][c] == 2:
                    traps += 1
                    
            return SearchResult("A*", clean_path, len(clean_path), expanded, elapsed, cost, True, traps)
        else:
            adjacent_nodes = env.get_neighbors(node)
            for node2 in adjacent_nodes:
                cost2 = env.step_cost(node2)
                new_path = path.copy()
                new_path.append((node2, cost2))
                queue.append(new_path)

    elapsed = time.perf_counter() - t0
    return SearchResult("A*", [], 0, expanded, elapsed, 0, False)



def risk_aware_a_star(env, risk_map):
    RISK_WEIGHT_G = 2.0
    RISK_WEIGHT_H = 3.0
    
    t0 = time.perf_counter()
    
    def path_f_cost(path):
        g_cost = 0
        for (node, cost) in path:
            g_cost += cost
        last_node = path[-1][0]
        
        h_cost = env.heuristic(last_node) + risk_map[last_node[0]][last_node[1]] * RISK_WEIGHT_H
        f_cost = g_cost + h_cost
        return f_cost, last_node
        
    visited = set()
    queue = [[(env.start, 0)]]
    expanded = 0
    
    while queue:
        queue.sort(key=path_f_cost)
        path = queue.pop(0)
        node = path[-1][0]
        
        if node in visited:
            continue
        visited.add(node)
        expanded += 1
        
        if node == env.goal:
            clean_path = []
            for p in path:
                clean_path.append(p[0])
                
            elapsed = time.perf_counter() - t0
            
            cost = 0
            for p in clean_path:
                cost += env.step_cost(p)
                
            traps = 0
            for p in clean_path:
                r = p[0]
                c = p[1]
                if env.grid[r][c] == 2:
                    traps += 1
                    
            return SearchResult("Risk-Aware A*", clean_path, len(clean_path), expanded, elapsed, cost, True, traps)
        else:
            adjacent_nodes = env.get_neighbors(node)
            for node2 in adjacent_nodes:
                risk_pen = risk_map[node2[0]][node2[1]] * RISK_WEIGHT_G
                cost2 = env.step_cost(node2) + risk_pen
                new_path = path.copy()
                new_path.append((node2, cost2))
                queue.append(new_path)

    elapsed = time.perf_counter() - t0
    return SearchResult("Risk-Aware A*", [], 0, expanded, elapsed, 0, False)
