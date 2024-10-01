# 8-puzzle

## Project Overview
This project implements an 8-puzzle solver that solves the classic 8-puzzle problem using multiple search algorithms. The 8-puzzle problem is a sliding puzzle consisting of a 3x3 grid, with eight numbered tiles and a blank space. The goal is to rearrange the tiles into a specific order by sliding the tiles into the blank space, starting from an initial configuration.

## Key Features
* Implements multiple search algorithms to solve the 8-puzzle problem:
  * Breadth-First Search (BFS): Explores all possible configurations level by level.
  * Uniform Cost Search (UCS): Explores the least-cost path to reach the goal configuration.
  * Depth-First Search (DFS): Explores possible configurations deeply before backtracking.
  * Depth-Limited Search (DLS): A variant of DFS with a specified depth limit.
  * Iterative Deepening Search (IDS): Combines the benefits of DFS and BFS by gradually increasing depth limits.
  * Greedy Search: Prioritizes moves that appear to be closer to the goal, using a heuristic.
  * A Search*: Combines path cost and heuristic for optimal searching, using Manhattan distance as the heuristic function.
#### Dump File Creation: 
For each search algorithm, the program creates a dump file to log the steps taken during the solution process. This includes:
  * The sequence of moves from the initial state to the goal state.
  * The number of steps, explored nodes, and time taken to reach the solution.



## Code Structure:
	-Node Class
	-Problem Class
	-Helper function (read from file, write trace to file and Manhattan distance)
	-Search Alorithms
		*BFS (uses dump_trace_bfs)
		*UCS (uses dump_trace_ucs)
		*DFS (uses dump_trace_depth_search)
		*DLS (uses dump_trace_depth_search)
		*IDS (uses dump_trace_depth_search)
		*Greedy (uses dump_trace_informed)
		*A star (uses dump_trace_informed)
	-Main Function

### To run the code:  
python expense_8_puzzle.py start-file.txt goal-file.txt breadth_first_search true

##### start-file.txt
contains initial state of puzzle

##### goal-file.txt
contains goal state of puzzle
