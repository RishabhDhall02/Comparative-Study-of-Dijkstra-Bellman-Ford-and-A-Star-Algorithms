# Comparative Study of Dijkstra, Bellman-Ford, and A* Algorithms

## Project Overview
This project explores classical graph algorithms for shortest path computation, focusing on performance comparison and real-world applications. The main algorithms studied are:

- **Dijkstra's Algorithm** – finds shortest paths on graphs with non-negative weights.
- **Bellman-Ford Algorithm** – finds shortest paths even with negative edge weights.
- **A\* Algorithm** – heuristic-driven pathfinding optimized for spatial networks.

The project benchmarks these algorithms using both randomly generated graphs and real-world data from the London Underground.

---

## Features
- Implementation of weighted graph structure (`WeightedGraph`) supporting node and edge operations.
- Multiple shortest path algorithm variants with:
  - Standard Dijkstra
  - Dijkstra with limited relaxations
  - Bellman-Ford for positive and negative weights
  - A* with customizable heuristics
- Real-world London Underground graph:
  - CSV parser for station data and connections
  - Haversine distance-based heuristics for spatial pathfinding
- Benchmarking of algorithm runtimes with Matplotlib visualizations
- Generation of random, good heuristic, and bad heuristic node pairs

---

## Dependencies
- Python 3.8+
- NumPy
- Matplotlib
- `csv`, `heapq`, `math`, `random` (built-in)

# File Structure
- `part1.py`: Comparison between Dijkstra's Algorithm and Bellman-Ford's Algorithm (based on randomly generated graphs)
- `part2.py`: Comparison between Dijkstra's Algorithm and A* Algorithm (based on the London Subway system)
