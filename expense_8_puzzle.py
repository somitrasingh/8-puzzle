import sys
import heapq
from collections import deque
from datetime import datetime


class Node:
    def __init__(self, state, parent=None, action=None, cost=0, depth=0, heuristic=0, comparison_mode='ucs'):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.depth = depth
        self.heuristic = heuristic
        self.comparison_mode = comparison_mode

    def __lt__(self, other):
        if self.comparison_mode == "ucs":
            return self.cost < other.cost
        elif self.comparison_mode == "greedy":
            return self.heuristic < other.heuristic
        elif self.comparison_mode == "a_star":
            return (self.cost + self.heuristic) < (other.cost + other.heuristic)
        return self.cost < other.cost




class Problem:
    def __init__(self, initial_state, goal_state):
        self.initial_state = initial_state
        self.goal_state = goal_state

    def is_goal(self, node):
        return node == self.goal_state

    def expand(self, node):
        children = []
        blank_pos = self.find_blank(node.state)

        if blank_pos is None:
            raise ValueError("No blank tile found in the state!")
        moves = self.possible_moves(blank_pos)

        for move in moves:
            child = [row[:] for row in node.state]
            new_blank = (blank_pos[0] + move[0], blank_pos[1] + move[1])
            child[blank_pos[0]][blank_pos[1]], child[new_blank[0]][new_blank[1]] = child[new_blank[0]][new_blank[1]], \
                child[blank_pos[0]][blank_pos[1]]

            tile_swapped = child[blank_pos[0]][blank_pos[1]]

            action = self.get_action(move, tile_swapped)

            cost = node.cost + tile_swapped

            child_state = Node(child, node, action, cost, node.depth+1, comparison_mode=node.comparison_mode)
            children.append(child_state)

        return children

    def find_blank(self, state):
        for i in range(len(state)):
            for j in range(len(state[i])):
                if state[i][j] == 0:
                    return i, j
        return None

    def possible_moves(self, blank_pos):

        moves = []
        if blank_pos[0] > 0:
            moves.append((-1, 0))
        if blank_pos[0] < 2:
            moves.append((1, 0))
        if blank_pos[1] > 0:
            moves.append((0, -1))
        if blank_pos[1] < 2:
            moves.append((0, 1))

        return moves

    def get_action(self, move, tile_swapped):
        if (move == (-1, 0)):
            action = f"Move {tile_swapped} Down"
        elif (move == (1, 0)):
            action = f"Move {tile_swapped} Up"
        elif (move == (0, -1)):
            action = f"Move {tile_swapped} Right"
        elif (move == (0, 1)):
            action = f"Move {tile_swapped} Left"
        return action

def read_from_file(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
        state = []
        for line in lines:
            line = line.strip()
            if line == "END OF FILE":
                break
            state.append(list(map(int, line.split())))

        return state
    
def manhattan_distance(state, goal_state):
    total_cost = 0
    for i in range(3):
        for j in range(3):
            tile = state[i][j]
            if tile != 0:
                # Find the position of this tile in the goal state
                for x in range(3):
                    for y in range(3):
                        if goal_state[x][y] == tile:
                            total_cost += tile*(abs(i - x) + abs(j - y))
                            break
    return total_cost



def write_trace_to_file(trace_steps, start_file, goal_file, method, dump_flag):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"trace-{current_time}.txt"
    with open(filename, 'w') as file:
        file.write(f"Command-Line Arguments : [{start_file}, {goal_file}, {method}, {dump_flag}]\n")
        file.write((f"Method Selected: {method}\n"))
        file.write(f"Running {method}\n\n\n\n")
        for line in trace_steps:
            file.write(line + '\n')
    print(f"Trace steps written to {filename}")


def dump_trace_bfs(node, children, fringe, closed, trace_steps):
    trace_steps.append(
        f"Generating successor of <state = {node.state}, action = ({node.action or 'Start'}), Parent = {node.parent.state if node.parent else 'None'}>\n")
    trace_steps.append(f"{len(children)} successors generated\n")
    trace_steps.append(f"Closed: [\n")
    for closed_states in closed:
        trace_steps.append(f"  {closed_states}\n")
    trace_steps.append(f"]\n")

    trace_steps.append(f"Fringe: [\n")
    for states in fringe:
        trace_steps.append(
            f"<state = {states.state}, action = ({states.action or 'Start'}), parent = {states.parent.state if states.parent else 'None'}\n")
    trace_steps.append(f"]\n")


def bfs(problem, dump_trace):
    nodes_popped = 0
    nodes_expanded = 0
    nodes_generated = 0
    max_fringe = 0
    trace_steps = []

    node = Node(problem.initial_state)

    if problem.is_goal(node.state):
        return node, nodes_popped, nodes_expanded, nodes_generated, max_fringe, trace_steps

    fringe = deque([node])
    closed = set()

    max_fringe = max(len(fringe), max_fringe)

    while fringe:
        node = fringe.popleft()
        nodes_popped += 1
        if problem.is_goal(node.state):
            return node, nodes_popped, nodes_expanded, nodes_generated, max_fringe, trace_steps
        if tuple(map(tuple, node.state)) in closed:
            continue
        closed.add(tuple(map(tuple, node.state)))

        children = problem.expand(node)

        if children:
            nodes_expanded += 1

        for child in children:
            nodes_generated += 1

            fringe.append(child)

        max_fringe = max(max_fringe, len(fringe))

        if dump_trace:
            dump_trace_bfs(node, children, fringe, closed, trace_steps)


def dump_trace_ucs(node, children, fringe, closed, trace_steps):
    trace_steps.append(
        f"Generating successor of <state = {node.state}, action = ({node.action or 'Start'}), Parent = {node.parent.state if node.parent else 'None'}, cost = {node.cost}>\n"
    )
    trace_steps.append(f"{len(children)} successors generated\n")
    trace_steps.append(f"Closed: [\n")
    for node, node_cost in closed.items():
        trace_steps.append(f" {node},  cost = {node_cost}\n")
    trace_steps.append(f"]\n")

    trace_steps.append(f"Fringe: [\n")
    for node_cost, node in fringe:
        trace_steps.append(
            f" {node.state}, action = ({node.action or 'Start'}), Parent = {node.parent.state if node.parent else 'None'} cost = {node_cost}\n")
    trace_steps.append(f"]\n")


def ucs(problem, dump_trace):
    nodes_popped = 0
    nodes_expanded = 0
    nodes_generated = 0
    max_fringe = 0
    trace_steps = []

    closed = {}
    node = Node(problem.initial_state)
    if problem.is_goal(node.state):
        return node, nodes_popped, nodes_expanded, nodes_generated, max_fringe, trace_steps

    fringe = [(node.cost, node)]

    max_fringe = max(len(fringe), max_fringe)

    while fringe:
        node_cost, node = heapq.heappop(fringe)
        nodes_popped += 1
        if problem.is_goal(node.state):
            return node, nodes_popped, nodes_expanded, nodes_generated, max_fringe, trace_steps

        state_tuple = tuple(map(tuple, node.state))
        if state_tuple in closed and closed[state_tuple] < node_cost:
            continue
        closed[state_tuple] = node_cost
        children = problem.expand(node)
        if children:
            nodes_expanded += 1

        for child in children:
            nodes_generated += 1
            heapq.heappush(fringe, (child.cost, child))

        max_fringe = max(max_fringe, len(fringe))

        if dump_trace:
            dump_trace_ucs(node, children, fringe, closed, trace_steps)


def dump_trace_depth_search(node, children, fringe, closed, trace_steps):
    trace_steps.append(
        f"Generating successor of <state = {node.state}, action = ({node.action or 'Start'}), Parent = {node.parent.state if node.parent else 'None'}, depth = {node.depth}>\n")

    trace_steps.append(f"{len(children)} successors generated\n")
    trace_steps.append(f"Closed: [\n")
    for closed_states in closed:
        trace_steps.append(f"    {closed_states}\n")
    trace_steps.append(f"]\n")

    trace_steps.append(f"Fringe: [\n")
    for states in fringe:
        trace_steps.append(
            f"<state = {states.state}, action = ({states.action or 'Start'}), parent = {states.parent.state if states.parent else 'None'}, depth = {states.depth}>\n")

    trace_steps.append(f"]\n")


def dfs(problem, dump_trace):
    nodes_popped = 0
    nodes_expanded = 0
    nodes_generated = 0
    max_fringe = 0
    trace_steps = []

    closed = set()
    node = Node(problem.initial_state)
    fringe = deque()
    if problem.is_goal(node.state):
        return node, nodes_popped, nodes_expanded, nodes_generated, max_fringe, trace_steps

    fringe.append(node)
    max_fringe = max(max_fringe, len(fringe))
    while fringe:
        node = fringe.pop()
        nodes_popped += 1

        if problem.is_goal(node.state):
            return node, nodes_popped, nodes_expanded, nodes_generated, max_fringe, trace_steps

        if tuple(map(tuple, node.state)) in closed:
            continue

        closed.add(tuple(map(tuple, node.state)))

        children = problem.expand(node)

        if children:
            nodes_expanded += 1

        for child in children:
            nodes_generated += 1
            fringe.append(child)

        max_fringe = max(max_fringe, len(fringe))

        if dump_trace:
            dump_trace_depth_search(node, children, fringe, closed, trace_steps)


def dls(problem, dump_trace, depth_limit):
    nodes_popped = 0
    nodes_expanded = 0
    nodes_generated = 0
    max_fringe = 0
    trace_steps = []

    closed = set()
    node = Node(problem.initial_state, depth=0)
    fringe = deque()

    if problem.is_goal(node.state):
        return node, nodes_popped, nodes_expanded, nodes_generated, max_fringe, trace_steps

    fringe.append(node)
    max_fringe = max(max_fringe, len(fringe))


    while fringe:
        node = fringe.pop()
        nodes_popped += 1

        if problem.is_goal(node.state):
            return node, nodes_popped, nodes_expanded, nodes_generated, max_fringe, trace_steps

        if tuple(map(tuple, node.state)) in closed:
            continue

        closed.add(tuple(map(tuple, node.state)))

        if node.depth < depth_limit:
            children = problem.expand(node)

            if children:
                nodes_expanded += 1

            for child in children:
                child.depth = node.depth + 1
                nodes_generated += 1
                fringe.append(child)

            max_fringe = max(max_fringe, len(fringe))
            if dump_trace:
                dump_trace_depth_search(node, children, fringe, closed, trace_steps)

    return None, nodes_popped, nodes_expanded, nodes_generated, max_fringe, trace_steps


def ids(problem, dump_trace):
    nodes_popped_total = 0
    nodes_expanded_total = 0
    nodes_generated_total = 0
    max_fringe_total = 0
    trace_steps = []
    depth_limit = 0

    while True:
        final_node, nodes_popped, nodes_expanded, nodes_generated, max_fringe, trace = dls(problem,
                                                                                                 dump_trace=dump_trace,
                                                                                                 depth_limit=depth_limit)
        nodes_popped_total += nodes_popped
        nodes_expanded_total += nodes_expanded
        nodes_generated_total += nodes_generated
        max_fringe_total = max(max_fringe_total, max_fringe)

        trace_steps.extend(trace)

        if final_node:
            return final_node, nodes_popped_total, nodes_expanded_total, nodes_generated_total, max_fringe_total, trace_steps

        depth_limit += 1
    return None, nodes_popped, nodes_expanded, nodes_generated, max_fringe, trace_steps




def dump_trace_informed(node, children, fringe, closed, trace_steps):
    trace_steps.append(
        f"Generating successor of <state = {node.state}, action = ({node.action or 'Start'}), Parent = {node.parent.state if node.parent else 'None'}, heuristic = {node.heuristic}>\n"
    )
    trace_steps.append(f"{len(children)} successors generated\n")
    trace_steps.append(f"Closed: [\n")
    for node in closed:
        trace_steps.append(f" {node}\n")
    trace_steps.append(f"]\n")

    trace_steps.append(f"Fringe: [\n")
    for node_cost, node in fringe:
        trace_steps.append(
            f" {node.state}, action = ({node.action or 'Start'}), Parent = {node.parent.state if node.parent else 'None'}, heuristic = {node.heuristic}\n")
    trace_steps.append(f"]\n")


def gs(problem, dump_trace):
    nodes_popped = 0
    nodes_expanded = 0
    nodes_generated = 0
    max_fringe = 0
    trace_steps = []

    node = Node(problem.initial_state, heuristic=manhattan_distance(problem.initial_state, problem.goal_state), comparison_mode="greedy")

    if problem.is_goal(node.state):
        return node, nodes_popped, nodes_expanded, nodes_generated, max_fringe, trace_steps

    fringe = [(node.heuristic, node)]
    closed = set()

    max_fringe = max(len(fringe), max_fringe)

    while fringe:
        _, node = heapq.heappop(fringe)
        nodes_popped += 1

        if problem.is_goal(node.state):
            return node, nodes_popped, nodes_expanded, nodes_generated, max_fringe, trace_steps

        if tuple(map(tuple, node.state)) in closed:
            continue

        closed.add(tuple(map(tuple, node.state)))

        children = problem.expand(node)

        if children:
            nodes_expanded += 1

        for child in children:
            child.heuristic = manhattan_distance(child.state, problem.goal_state)
            nodes_generated += 1
            heapq.heappush(fringe, (child.heuristic, child))

        max_fringe = max(max_fringe, len(fringe))

        if dump_trace:
            dump_trace_informed(node, children, fringe, closed, trace_steps)

    return None, nodes_popped, nodes_expanded, nodes_generated, max_fringe, trace_steps


def dump_trace_astar(node, children, fringe, closed, trace_steps):
    trace_steps.append(
        f"Generating successor of <state = {node.state}, action = ({node.action or 'Start'}), Parent = {node.parent.state if node.parent else 'None'}, f(n) = {node.heuristic + node.cost}>\n"
    )
    trace_steps.append(f"{len(children)} successors generated\n")
    trace_steps.append(f"Closed: [\n")
    for node in closed:
        trace_steps.append(f" {node}\n")
    trace_steps.append(f"]\n")

    trace_steps.append(f"Fringe: [\n")
    for node_cost, node in fringe:
        trace_steps.append(
            f" {node.state}, action = ({node.action or 'Start'}), Parent = {node.parent.state if node.parent else 'None'}, f(n) = {node.heuristic + node.cost}\n")
    trace_steps.append(f"]\n")


def a_star(problem, dump_trace):
    nodes_popped = 0
    nodes_expanded = 0
    nodes_generated = 0
    max_fringe = 0
    trace_steps = []

    fringe = []

    closed = {}

    node = Node(problem.initial_state, heuristic=manhattan_distance(problem.initial_state, problem.goal_state), comparison_mode="a_star")

    heapq.heappush(fringe, (node.cost + node.heuristic, node))
    max_fringe = max(len(fringe), max_fringe)

    while fringe:
        _, node = heapq.heappop(fringe)
        nodes_popped += 1

        if problem.is_goal(node.state):
            return node, nodes_popped, nodes_expanded, nodes_generated, max_fringe, trace_steps

        state_tuple = tuple(map(tuple, node.state))

        if state_tuple in closed and closed[state_tuple] <= node.cost:
            continue

        closed[state_tuple] = node.cost

        children = problem.expand(node)
        if children:
            nodes_expanded += 1

        for child in children:
            child.heuristic = manhattan_distance(child.state, problem.goal_state)
            nodes_generated += 1
            heapq.heappush(fringe, (child.cost + child.heuristic, child))

        max_fringe = max(max_fringe, len(fringe))

        if dump_trace:
            dump_trace_informed(node, children, fringe, closed, trace_steps)

    return None, nodes_popped, nodes_expanded, nodes_generated, max_fringe, trace_steps


def main():
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: python expense_8_puzzle.py <start-file> <goal-file> <method> <dump-flag>")
        return

    start_file = sys.argv[1]
    goal_file = sys.argv[2]
    method = sys.argv[3]
    dump_flag = False

    if len(sys.argv) == 5 and sys.argv[4].lower() == 'true':
        dump_flag = True


    initial_state = read_from_file(start_file)
    goal_state = read_from_file(goal_file)

    problem = Problem(initial_state, goal_state)

    if method == 'breadth_first_search':
        final_node, nodes_popped, nodes_expanded, nodes_generated, max_fringe, trace_steps = bfs(problem,
                                                                                                 dump_trace=dump_flag)
    elif method == 'uniform_cost_search':
        final_node, nodes_popped, nodes_expanded, nodes_generated, max_fringe, trace_steps = ucs(problem,
                                                                                                 dump_trace=dump_flag)
    elif method == 'depth_first_search':
        final_node, nodes_popped, nodes_expanded, nodes_generated, max_fringe, trace_steps = dfs(problem,
                                                                                                 dump_trace=dump_flag)
    elif method == 'depth_limited_search':
        depth_limit = 30
        final_node, nodes_popped, nodes_expanded, nodes_generated, max_fringe, trace_steps = dls(problem,
                                                                                                 dump_trace=dump_flag, depth_limit=depth_limit)
    elif method == 'iterative_deepening_search':
        final_node, nodes_popped, nodes_expanded, nodes_generated, max_fringe, trace_steps = ids(problem, dump_trace=dump_flag)

    elif method == 'greedy_search':
        final_node, nodes_popped, nodes_expanded, nodes_generated, max_fringe, trace_steps = gs(problem, dump_trace=dump_flag)

    else:
        final_node, nodes_popped, nodes_expanded, nodes_generated, max_fringe, trace_steps = a_star(problem,
                                                                                                dump_trace=dump_flag)


    if final_node:
        path = []
        moves = []
        total_cost = final_node.cost
        depth = final_node.depth
        node = final_node
        while node:
            path.append(node.state)
            if node.action:
                moves.append(node.action)
            node = node.parent

        path.reverse()
        moves.reverse()

        for move in moves:
            print(move)

        print(f"\n Nodes Popped: {nodes_popped}")
        print(f" Nodes Expanded: {nodes_expanded}")
        print(f" Nodes Generated: {nodes_generated}")
        print(f" Max Fringe Size: {max_fringe}")
        print(f" Solution found at {depth} with cost {total_cost}.")

        if dump_flag:
            write_trace_to_file(trace_steps, start_file, goal_file, method, dump_flag)
    else:
        print("No solution found")


if __name__ == "__main__":
    main()



