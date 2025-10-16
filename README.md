# Comparative Study of Dijkstra, Bellman-Ford, and A* Algorithms

## About
This repository is an academic project showcasing a comparative study of shortest-path algorithms, including Dijkstra, Bellman-Ford, and A*. It demonstrates the implementation, analysis, and performance evaluation of these algorithms using both synthetic and real-world datasets (e.g., the London Subway system).  

This project is intended as a demonstration of algorithm design and analysis for educational purposes. The Python scripts provided can be run directly, but this is **not** a production-ready library and does not include installation or package support.

## Project Overview
This project explores classical graph algorithms for shortest path computation, focusing on performance comparison and real-world applications. The main algorithms studied are:

- **Dijkstra's Algorithm** – finds shortest paths on graphs with non-negative weights.
- **Bellman-Ford Algorithm** – finds shortest paths even with negative edge weights.
- **A\* Algorithm** – heuristic-driven pathfinding optimized for spatial networks.

The project benchmarks these algorithms using both randomly generated graphs and real-world data from the London Subway system.

---

## Features
- Implementation of weighted graph structure (`WeightedGraph`) supporting node and edge operations.
- Multiple shortest path algorithm variants with:
  - Standard Dijkstra
  - Dijkstra with limited relaxations
  - Bellman-Ford for positive and negative weights
  - A* with customizable heuristics
- Real-world London Subway system graph:
  - CSV parser for station data and connections
  - Haversine distance-based heuristics for spatial pathfinding
- Benchmarking of algorithm runtimes with Matplotlib visualizations
- Generation of random, good heuristic, and bad heuristic node pairs

---

## Project Requirements
- Python 3.8+
- NumPy
- Matplotlib
- `csv`, `heapq`, `math`, `random` (built-in)

---

## File Structure
- `part1.py`: Comparison between Dijkstra's Algorithm and Bellman-Ford's Algorithm (based on randomly generated graphs)
- `part2.py`: Comparison between Dijkstra's Algorithm and A* Algorithm (based on the London Subway system)
- `london_stations.csv`: Station IDs, names, latitudes, and longitudes
- `london_connections.csv`: Edges between stations

---

## Key Insights
- Part 1 experiments show that modifying Dijkstra's and Bellman-Ford's algorithms to limit node relaxations (using a parameter `k`) can improve runtime, but smaller `k` values may reduce accuracy.
- On average, Dijkstra’s algorithm is faster than Bellman-Ford’s, though Bellman-Ford can sometimes outperform Dijkstra on sparse graphs.
- For all-pairs shortest paths, Dijkstra’s handles positive-weight graphs efficiently, while Bellman-Ford is necessary for graphs with negative weights.
- Part 2 experiments using the London Subway system demonstrate how A*’s heuristic can greatly improve time complexity for specific source-destination queries.
- A* is generally faster than Dijkstra for targeted pathfinding, but Dijkstra is more consistent and predictable since it computes distances to all nodes.
- Algorithm performance depends on multiple factors: graph density, number of nodes/edges, presence of negative weights, and choice of heuristic.
- Choosing the appropriate shortest-path algorithm requires balancing speed, accuracy, and suitability for the specific graph scenario.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or feedback, please contact [Rishabh Dhall](mailto:rishabhdhall02@gmail.com).
