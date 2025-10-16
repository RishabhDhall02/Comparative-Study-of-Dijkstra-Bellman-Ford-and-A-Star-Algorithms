# Comparative Study of Dijkstra, Bellman-Ford, and A*

## Project Overview
This project is an in-depth exploration of classical graph algorithms for shortest path computation. It focuses on:

- **Dijkstra's Algorithm** – efficient for positive-weighted graphs.
- **Bellman-Ford Algorithm** – handles negative weights and detects negative cycles.
- **A\* Algorithm** – heuristic-driven pathfinding, optimized for spatial graphs (e.g., London Subway).

The goal is to compare the performance, behavior, and accuracy of these algorithms under different scenarios, including:

- Random graphs with positive weights.
- Real-world spatial networks (London subway system).
- Varying heuristics in A\* to test efficiency and optimality.

## Features
- Weighted graph representation (undirected graphs).
- Support for positive and negative edge weights.
- Multiple variations of Dijkstra and Bellman-Ford (limited relaxation, all-pairs).
- A\* implementation with customizable heuristics.
- Real-world London subway dataset analysis.
- Runtime benchmarking and visualizations using Matplotlib.

## Dependencies
- Python 3.8+
- NumPy
- Matplotlib
- CSV module (built-in)
- `heapq` and `random` (built-in)


