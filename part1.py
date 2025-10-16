import heapq
import random
import timeit
import numpy as np
import matplotlib.pyplot as plt

# Function for plotting the graph and saving the image as well
def draw_plot(arr1, arr2, mean1, mean2, filename):
    x = np.arange(0, len(arr1), 1)
    fig = plt.figure(figsize=(20, 8))
    plt.plot(x, arr1, label=f"Dikstra's (Avg: {mean1:.2e})", color="maroon", linestyle='-', marker='o')
    plt.plot(x, arr2, label=f"Bellman-Ford's (Avg: {mean2:.2e})", color="blue", linestyle='-', marker='o')
    plt.xlabel("Iterations")
    plt.ylabel("Run time in ms order of 1e-6")
    plt.title("Run time for retrieval")
    plt.legend()
    plt.savefig(filename, dpi=300, bbox_inches = "tight")
    plt.show()

class WeightedGraph:
    def __init__(self, nodes):
        self.graph = {}
        self.weight = {}
        for i in range(nodes):
            self.graph[i] = []

    def are_connected(self, node1, node2):
        for node in self.graph[node1]:
            if node == node2:
                return True
        return False

    def connected_nodes(self, node):
        return self.graph[node]

    def add_node(self,):
        #add a new node number = length of existing node
        self.graph[len(self.graph)] = []

    def add_edge(self, node1, node2, weight):
        if node1 not in self.graph[node2]:
            self.graph[node1].append(node2)
            self.weight[(node1, node2)] = weight

            #since it is undirected
            self.graph[node2].append(node1)
            self.weight[(node2, node1)] = weight

    def number_of_nodes(self,):
        return len(self.graph)

    def has_edge(self, src, dst):
        return dst in self.graph[src] 

    def get_weight(self,):
        total = 0
        for node1 in self.graph:
            for node2 in self.graph[node1]:
                total += self.weight[(node1, node2)]
           
        # because it is undirected
        return total/2

##variation of Dijkstra's
def dijkstra(graph: WeightedGraph, source, k):
    distances = {}
    paths = {}
    relaxationCounter = {}
    ##checks for 0 < k < N - 1
    if (k >= graph.number_of_nodes() - 1):
        raise ValueError("k must be lesser than N (number of nodes) - 1.")
    if (k <= 0):
        raise ValueError("k must be greater than 0.")
    ##initialize distances, paths, and relaxation counter
    for node in graph.graph:
        distances[node] = float('inf')
        paths[node] = []
        relaxationCounter[node] = 0
    distances[source] = 0
    paths[source] = [source]
    ##initialize priority queue starting at source node
    pq = [(0, source)]
    while pq:
        currentDistance, currentNode = heapq.heappop(pq)
        ##relax if node hasn't hit relaxation limit
        if relaxationCounter[currentNode] < k:
            ##explore all neighbors of the current node (relaxation)
            for neighbor in graph.graph[currentNode]:
                potentialDistance = currentDistance + graph.weight.get((currentNode, neighbor), float('inf'))
                ##if a shorter path to the neighbor is found relax the edge
                if potentialDistance < distances[neighbor]:
                    distances[neighbor] = potentialDistance
                    paths[neighbor] = paths[currentNode] + [neighbor]
                    relaxationCounter[neighbor] += 1
                    heapq.heappush(pq, (potentialDistance, neighbor))
    return distances, paths

##variation of bellman ford
def bellman_ford(graph: WeightedGraph, source, k):
    distances = {}
    paths = {}
    relaxationCounter = {}
    ##checks for 0 < k < N - 1
    if (k >= graph.number_of_nodes() - 1):
        raise ValueError("k must be lesser than N (number of nodes) - 1.")
    if (k <= 0):
        raise ValueError("k must be greater than 0.")
    ##initialize distances, paths, and relaxation counter
    for node in graph.graph:
        distances[node] = float('inf')
        paths[node] = []
        relaxationCounter[node] = 0
    distances[source] = 0
    paths[source] = [source]
    ##relax edges up to k times
    for i in range(k):
        updated = False
        newDistances = distances.copy()
        newPaths = paths.copy()
        for u in graph.graph:
            for v in graph.graph[u]:
                ##calculate the potential new distance between two nodes
                potentialDistance = distances[u] + graph.weight.get((u, v), float('inf'))
                ##if distance is smaller than current distance and relax cap hasnt been reached then relax
                if (potentialDistance < newDistances[v]) and (relaxationCounter[v] < k):
                    newDistances[v] = potentialDistance
                    newPaths[v] = paths[u] + [v]
                    relaxationCounter[v] += 1
                    updated = True
        ##iteration didn't change
        if updated == False:
            break
        ##update distances and paths
        distances = newDistances
        paths = newPaths
    return distances, paths

##helper function to create randomized graph
def generateRandomGraph(nodes, edges, weightRangeMin, weightRangeMax):
    ##check validity of edge numbers
    if edges > (nodes * (nodes - 1) // 2):
        print("number of edges to nodes is invalid")
        return False
    graph = WeightedGraph(nodes)
    ##track which nodes have edges added already
    edgesAdded = set()
    ##add required number of edges randomly
    while len(edgesAdded)//2 < edges:
        node1 = random.randint(0, nodes - 1)
        node2 = random.randint(0, nodes - 1)
        ##add edge if edge not already exist and nodes are different
        if (node1 != node2) and ((node1, node2) not in edgesAdded):
            ##generate edge weight
            weight = random.randint(weightRangeMin, weightRangeMax)
            graph.add_edge(node1, node2, weight)
            edgesAdded.add((node1, node2))
            edgesAdded.add((node2, node1))
    return graph

##helper function to print list for progress checking
def printList(listt):
    for i in listt:
        print(i)
    print("\n")

def part2experiment():
    ##run with various graph densities and k values
    source = 0
    nodes = 30
    edges = [30,230,430]
    weightRangeMin = 1
    weightRangeMax = 20
    kValues = [1,15,28]
    dijkstrasResult = []
    bellmandResult = []
    dijkstrasTotal = 0
    bellmandTotal = 0
    ##testing the k value of 1 with each of the 3 edges
    for i in range(80):
        randomGraph = generateRandomGraph(nodes, edges[0], weightRangeMin, weightRangeMax)
        startTime = timeit.default_timer()
        dijkstra(randomGraph, source, kValues[0])
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        dijkstrasResult.append(elapsedTime)
        dijkstrasTotal += elapsedTime
        startTime = timeit.default_timer()
        bellman_ford(randomGraph, source, kValues[0])
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        bellmandResult.append(elapsedTime)
        bellmandTotal += elapsedTime
    draw_plot(dijkstrasResult, bellmandResult, dijkstrasTotal/80, bellmandTotal/80, "Edge_0_K_0.png")
    dijkstrasResult = []
    bellmandResult = []
    dijkstrasTotal = 0
    bellmandTotal = 0
    for i in range(80):
        randomGraph = generateRandomGraph(nodes, edges[0], weightRangeMin, weightRangeMax)
        startTime = timeit.default_timer()
        dijkstra(randomGraph, source, kValues[1])
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        dijkstrasResult.append(elapsedTime)
        dijkstrasTotal += elapsedTime
        startTime = timeit.default_timer()
        bellman_ford(randomGraph, source, kValues[1])
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        bellmandResult.append(elapsedTime)
        bellmandTotal += elapsedTime
    draw_plot(dijkstrasResult, bellmandResult, dijkstrasTotal/80, bellmandTotal/80, "Edge_0_K_1.png")     
    dijkstrasResult = []
    bellmandResult = []
    dijkstrasTotal = 0
    bellmandTotal = 0
    for i in range(80):
        randomGraph = generateRandomGraph(nodes, edges[0], weightRangeMin, weightRangeMax)
        startTime = timeit.default_timer()
        dijkstra(randomGraph, source, kValues[2])
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        dijkstrasResult.append(elapsedTime)
        dijkstrasTotal += elapsedTime
        startTime = timeit.default_timer()
        bellman_ford(randomGraph, source, kValues[2])
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        bellmandResult.append(elapsedTime)
        bellmandTotal += elapsedTime
    draw_plot(dijkstrasResult, bellmandResult, dijkstrasTotal/80, bellmandTotal/80, "Edge_0_K_2.png")
    dijkstrasResult = []
    bellmandResult = []
    dijkstrasTotal = 0
    bellmandTotal = 0
    ##testing the k value of 15 with each of the 3 edges
    for i in range(80):
        randomGraph = generateRandomGraph(nodes, edges[1], weightRangeMin, weightRangeMax)
        startTime = timeit.default_timer()
        dijkstra(randomGraph, source, kValues[0])
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        dijkstrasResult.append(elapsedTime)
        dijkstrasTotal += elapsedTime
        startTime = timeit.default_timer()
        bellman_ford(randomGraph, source, kValues[0])
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        bellmandResult.append(elapsedTime)
        bellmandTotal += elapsedTime
    draw_plot(dijkstrasResult, bellmandResult, dijkstrasTotal/80, bellmandTotal/80, "Edge_1_K_0.png")
    dijkstrasResult = []
    bellmandResult = []
    dijkstrasTotal = 0
    bellmandTotal = 0
    for i in range(80):
        randomGraph = generateRandomGraph(nodes, edges[1], weightRangeMin, weightRangeMax)
        startTime = timeit.default_timer()
        dijkstra(randomGraph, source, kValues[1])
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        dijkstrasResult.append(elapsedTime)
        dijkstrasTotal += elapsedTime
        startTime = timeit.default_timer()
        bellman_ford(randomGraph, source, kValues[1])
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        bellmandResult.append(elapsedTime)
        bellmandTotal += elapsedTime
    draw_plot(dijkstrasResult, bellmandResult, dijkstrasTotal/80, bellmandTotal/80, "Edge_1_K_1.png")
    dijkstrasResult = []
    bellmandResult = []
    dijkstrasTotal = 0
    bellmandTotal = 0
    for i in range(80):
        randomGraph = generateRandomGraph(nodes, edges[1], weightRangeMin, weightRangeMax)
        startTime = timeit.default_timer()
        dijkstra(randomGraph, source, kValues[2])
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        dijkstrasResult.append(elapsedTime)
        dijkstrasTotal += elapsedTime
        startTime = timeit.default_timer()
        bellman_ford(randomGraph, source, kValues[2])
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        bellmandResult.append(elapsedTime)
        bellmandTotal += elapsedTime
    draw_plot(dijkstrasResult, bellmandResult, dijkstrasTotal/80, bellmandTotal/80, "Edge_1_K_2.png")
    dijkstrasResult = []
    bellmandResult = []
    dijkstrasTotal = 0
    bellmandTotal = 0
    ##testing the k value of 28 with each of the 3 edges
    for i in range(80):
        randomGraph = generateRandomGraph(nodes, edges[2], weightRangeMin, weightRangeMax)
        startTime = timeit.default_timer()
        dijkstra(randomGraph, source, kValues[0])
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        dijkstrasResult.append(elapsedTime)
        dijkstrasTotal += elapsedTime
        startTime = timeit.default_timer()
        bellman_ford(randomGraph, source, kValues[0])
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        bellmandResult.append(elapsedTime)
        bellmandTotal += elapsedTime
    draw_plot(dijkstrasResult, bellmandResult, dijkstrasTotal/80, bellmandTotal/80, "Edge_2_K_0.png")
    dijkstrasResult = []
    bellmandResult = []
    dijkstrasTotal = 0
    bellmandTotal = 0
    for i in range(80):
        randomGraph = generateRandomGraph(nodes, edges[2], weightRangeMin, weightRangeMax)
        startTime = timeit.default_timer()
        dijkstra(randomGraph, source, kValues[1])
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        dijkstrasResult.append(elapsedTime)
        dijkstrasTotal += elapsedTime
        startTime = timeit.default_timer()
        bellman_ford(randomGraph, source, kValues[1])
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        bellmandResult.append(elapsedTime)
        bellmandTotal += elapsedTime
    draw_plot(dijkstrasResult, bellmandResult, dijkstrasTotal/80, bellmandTotal/80, "Edge_2_K_1.png")
    dijkstrasResult = []
    bellmandResult = []
    dijkstrasTotal = 0
    bellmandTotal = 0
    for i in range(80):
        randomGraph = generateRandomGraph(nodes, edges[2], weightRangeMin, weightRangeMax)
        startTime = timeit.default_timer()
        dijkstra(randomGraph, source, kValues[2])
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        dijkstrasResult.append(elapsedTime)
        dijkstrasTotal += elapsedTime
        startTime = timeit.default_timer()
        bellman_ford(randomGraph, source, kValues[2])
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        bellmandResult.append(elapsedTime)
        printList(dijkstrasResult)
        bellmandTotal += elapsedTime
    draw_plot(dijkstrasResult, bellmandResult, dijkstrasTotal/80, bellmandTotal/80, "Edge_2_K_2.png")
    dijkstrasResult = []
    bellmandResult = []
    dijkstrasTotal = 0
    bellmandTotal = 0
    ##testing different number of nodes: 30, 50, 70 with (N - 2) k values and proportionally same number of edges (2 x number of nodes)
    for i in range(80):
        randomGraph = generateRandomGraph(30, 60, weightRangeMin, weightRangeMax)
        startTime = timeit.default_timer()
        dijkstra(randomGraph, source, 28)
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        dijkstrasResult.append(elapsedTime)
        dijkstrasTotal += elapsedTime
        startTime = timeit.default_timer()
        bellman_ford(randomGraph, source, 28)
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        bellmandResult.append(elapsedTime)
        bellmandTotal += elapsedTime
    draw_plot(dijkstrasResult, bellmandResult, dijkstrasTotal/80, bellmandTotal/80, "Nodes_30.png")
    dijkstrasResult = []
    bellmandResult = []
    dijkstrasTotal = 0
    bellmandTotal = 0
    for i in range(80):
        randomGraph = generateRandomGraph(50, 100, weightRangeMin, weightRangeMax)
        startTime = timeit.default_timer()
        dijkstra(randomGraph, source, 48)
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        dijkstrasResult.append(elapsedTime)
        dijkstrasTotal += elapsedTime
        startTime = timeit.default_timer()
        bellman_ford(randomGraph, source, 48)
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        bellmandResult.append(elapsedTime)
        bellmandTotal += elapsedTime
    draw_plot(dijkstrasResult, bellmandResult, dijkstrasTotal/80, bellmandTotal/80, "Nodes_50.png")
    dijkstrasResult = []
    bellmandResult = []
    dijkstrasTotal = 0
    bellmandTotal = 0
    for i in range(80):
        randomGraph = generateRandomGraph(70, 140, weightRangeMin, weightRangeMax)
        startTime = timeit.default_timer()
        dijkstra(randomGraph, source, 68)
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        dijkstrasResult.append(elapsedTime)
        dijkstrasTotal += elapsedTime
        startTime = timeit.default_timer()
        bellman_ford(randomGraph, source, 68)
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        bellmandResult.append(elapsedTime)
        bellmandTotal += elapsedTime
    draw_plot(dijkstrasResult, bellmandResult, dijkstrasTotal/80, bellmandTotal/80, "Nodes_70.png")
    dijkstrasResult = []
    bellmandResult = []
    dijkstrasTotal = 0
    bellmandTotal = 0

part2experiment()