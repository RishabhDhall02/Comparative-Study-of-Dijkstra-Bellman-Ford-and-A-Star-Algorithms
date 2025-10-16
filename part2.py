import csv
import math
import os
import timeit
import numpy as np
import matplotlib.pyplot as plt
import heapq
import random

# Function that parses the london_stations.csv file and initialises the variables
def parse_london_stations(file_path):
    stations = {}
    with open(file_path, newline = '', encoding = 'utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            station_id = row['id']
            stations[station_id] = {
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
                'name': row['name']
            }
    return stations

# Function that parses the london_connections.csv file and initialises the variables
def parse_london_connections(file_path):
    connections = []
    with open(file_path, newline = '', encoding = 'utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            station1 = row['station1']
            station2 = row['station2']
            connections.append((station1, station2))
    return connections

# Graph class that creates the entire map of the London subway
class LondonGraph:
    def __init__(self):
        self.graph = {}
        self.locations = {}

    # Function for adding a new station (node)
    def add_station(self, station_id, latitude, longitude):
        self.graph[station_id] = []
        self.locations[station_id] = (latitude, longitude)
    
    # Function for adding a new connection between 2 stations (an edge between 2 nodes)
    def add_connection(self, station1, station2):
        if station1 in self.graph and station2 in self.graph:
            lat1, lon1 = self.locations[station1]
            lat2, lon2 = self.locations[station2]

            radius = 6371 # Earth's radius in km

            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

            self.graph[station1].append((station2, radius*c))
            self.graph[station2].append((station1, radius*c))

    # Uses the Haversine formula to find the actual distance between 2 stations
    def heuristic(self, station1, station2):
        lat1, lon1 = self.locations[station1]
        lat2, lon2 = self.locations[station2]

        radius = 6371 # Earth's radius in km

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
        return radius * c  # Distance in km

    # Simply gives 0 for the heuristic value - used for testing out if A* will function similar to Dijkstra's when the value is 0
    def zeroHeuristic(self, station1, station2):
        return 0
    
    # Returns the graph for debugging
    def get_graph(self,):
        return self.graph

stations = parse_london_stations('london_stations.csv')
connections = parse_london_connections('london_connections.csv')

london_graph = LondonGraph()

# Inputs all the nodes from the 'london_stations.csv' file into the graph
for station_id, data in stations.items():
    london_graph.add_station(station_id, data['latitude'], data['longitude'])

# Inputs all the edges from the 'london_connections.csv' file into the graph
for station1, station2 in connections:
    london_graph.add_connection(station1, station2)

##variation of Dijkstra's
def dijkstra(graph, source, destination):
    distances = {}
    paths = {}
    ##initialize distances, paths, and relaxation counter
    for node in graph.graph:
        distances[node] = float('inf')
        paths[node] = []
    distances[source] = 0
    paths[source] = [source]
    ##initialize priority queue starting at source node
    pq = [(0, source)]
    while pq:
        currentDistance, currentNode = heapq.heappop(pq)
        ##we reached distance
        if currentNode == destination:
            break
        ##explore all neighbors of the current node (relaxation)
        for neighbor, distance in graph.graph[currentNode]:
            potentialDistance = currentDistance + distance
            ##if a shorter path to the neighbor is found relax the edge
            if potentialDistance < distances[neighbor]:
                distances[neighbor] = potentialDistance
                paths[neighbor] = paths[currentNode] + [neighbor]
                heapq.heappush(pq, (potentialDistance, neighbor))
    return distances, paths

##helper function for A*
def reconstruct_path(predecessorMap, source, destination):
    """Reconstructs the shortest path from source to destination."""
    path = []
    current = destination

    while current in predecessorMap:
        path.append(current)
        current = predecessorMap[current]

    path.append(source)
    path.reverse()  # Reverse the path to get it from source to destination
    return path

##A* algorithm
def A_Star(graph, source, destination, heuristic):

    # Priority queue (min-heap) for A* search
    priorityQueue = []

    # Initialize gScore for all nodes using a loop
    # "Initialize gScore for all nodes"
    gScore = {}
    for node in graph.graph:
        gScore[node] = float('inf')  # Set initial gScore to infinity
    gScore[source] = 0  # The cost of reaching the source is 0

    # Push the source node into the queue with its fScore
    # "Use priority queue (heapq)"
    heapq.heappush(priorityQueue, (heuristic(source, destination), 0, source))  
    predecessorMap = {}

    while priorityQueue:

        # Pop the node with the lowest fScore
        currentFScore, currentGScore, currentNode = heapq.heappop(priorityQueue)

        # If we reached the destination, reconstruct and return the path
        # "Return the 2-tuple (predecessorMap, shortestPath)
        if currentNode == destination:
            path = reconstruct_path(predecessorMap, source, destination)
            return predecessorMap, path  

        # Explore neighbors
        for neighbor, time in graph.graph[currentNode]:
            newGScore = gScore[currentNode] + time

            if newGScore < gScore.get(neighbor, float('inf')):
                # Update predecessor and gScore
                predecessorMap[neighbor] = currentNode
                gScore[neighbor] = newGScore

                # Compute the estimated total cost (fScore)
                fScore = newGScore + heuristic(neighbor, destination)
               
                # Push the updated node into the priority queue
                heapq.heappush(priorityQueue, (fScore, newGScore, neighbor))

    # No path found
    return {}, []  # Return None and empty list if no path is found

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

##helper function to return two random nodes
def randomNodes():
    ##get all valid nodes from the station dictionary
    validNodes = list(stations.keys())
    ##randomly generate 2 nodes
    node1, node2 = random.sample(validNodes, 2)
    return node1, node2

##helper function to generate pairs of stations with good heuristic
def generateGoodPairs(graph):
    goodPairs = []
    stations = list(graph.locations.keys())
    for i in range(len(stations)):
        for j in range(i + 1, len(stations)):
            station1 = stations[i]
            station2 = stations[j]
            heuristicDistance = graph.heuristic(station1, station2)
            distancee, pathh = dijkstra(graph, station1, station2)
            realDistance = distancee[station2] 
            if realDistance > heuristicDistance:
                goodPairs.append((station1, station2))
    return goodPairs
    
##helper function to generate pairs of stations with bad heuristic
def generateBadPairs(graph):
    badPairs = []
    stations = list(graph.locations.keys())
    for i in range(len(stations)):
        for j in range(i + 1, len(stations)):
            station1 = stations[i]
            station2 = stations[j]
            heuristicDistance = graph.heuristic(station1, station2)*2
            distancee, pathh = dijkstra(graph, station1, station2)
            realDistance = distancee[station2]
            if realDistance < heuristicDistance:
                badPairs.append((station1, station2))
    return badPairs
    
##experiment
def experiment5():
    ##measure time complexity
    goodPairs = generateGoodPairs(london_graph)
    dijkstrasResult = []
    aStarResult = []
    dijkstrasTotal = 0
    aStarTotal = 0
    ##testing dikstras and A* with varying destination and sources but heuristic is 0
    for i in range(80):
        source, destination = randomNodes()
        startTime = timeit.default_timer()
        dijkstra(london_graph, source, destination)
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        dijkstrasResult.append(elapsedTime)
        dijkstrasTotal += elapsedTime
        startTime = timeit.default_timer()
        A_Star(london_graph, source, destination, london_graph.zeroHeuristic)
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        aStarResult.append(elapsedTime)
        aStarTotal += elapsedTime
    draw_plot(dijkstrasResult, aStarResult, dijkstrasTotal/80, aStarTotal/80, "Zero_Heuristic.png")
    dijkstrasResult = []
    aStarResult = []
    dijkstrasTotal = 0
    aStarTotal = 0
    ##testing dikstras and A* with good heuristic
    for i in range(80):
        source, destination = random.choice(goodPairs)
        startTime = timeit.default_timer()
        dijkstra(london_graph, source, destination)
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        dijkstrasResult.append(elapsedTime)
        dijkstrasTotal += elapsedTime
        startTime = timeit.default_timer()
        A_Star(london_graph, source, destination, london_graph.heuristic)
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        aStarResult.append(elapsedTime)
        aStarTotal += elapsedTime
    draw_plot(dijkstrasResult, aStarResult, dijkstrasTotal/80, aStarTotal/80, "Good_Heuristic.png")
    dijkstrasResult = []
    aStarResult = []
    dijkstrasTotal = 0
    aStarTotal = 0
    ##testing dikstras and A* with varying destination and sources (average case)
    for i in range(80):
        source, destination = randomNodes()
        startTime = timeit.default_timer()
        dijkstra(london_graph, source, destination)
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        dijkstrasResult.append(elapsedTime)
        dijkstrasTotal += elapsedTime
        startTime = timeit.default_timer()
        A_Star(london_graph, source, destination, london_graph.heuristic)
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        aStarResult.append(elapsedTime)
        aStarTotal += elapsedTime
    draw_plot(dijkstrasResult, aStarResult, dijkstrasTotal/80, aStarTotal/80, "Varying_Nodes.png")
    dijkstrasResult = []
    aStarResult = []
    dijkstrasTotal = 0
    aStarTotal = 0
    ##testing dikstras and A* with bad heuristic
    badPairs = generateBadPairs(london_graph)
    for i in range(80):
        source, destination = random.choice(badPairs)
        startTime = timeit.default_timer()
        dijkstra(london_graph, source, destination)
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        dijkstrasResult.append(elapsedTime)
        dijkstrasTotal += elapsedTime
        startTime = timeit.default_timer()
        A_Star(london_graph, source, destination, london_graph.heuristic)
        endTime = timeit.default_timer()
        elapsedTime = endTime - startTime
        aStarResult.append(elapsedTime)
        aStarTotal += elapsedTime
    draw_plot(dijkstrasResult, aStarResult, dijkstrasTotal/80, aStarTotal/80, "Bad_Heuristic.png")
    ##other tests
    source = '1'
    destination = '3'
    dijkstra(london_graph, source, destination)
    A_Star(london_graph, source, destination, london_graph.heuristic)
    source = '1'
    destination = '20'
    dijkstra(london_graph, source, destination)
    A_Star(london_graph, source, destination, london_graph.heuristic)
    source = '12'
    destination = '100'
    dijkstra(london_graph, source, destination)
    A_Star(london_graph, source, destination, london_graph.heuristic)
    source = '50'
    destination = '303'
    dijkstra(london_graph, source, destination)
    A_Star(london_graph, source, destination, london_graph.heuristic)
    source = '82'
    destination = '205'
    dijkstra(london_graph, source, destination)
    A_Star(london_graph, source, destination, london_graph.heuristic)
    source = '275'
    destination = '31'
    dijkstra(london_graph, source, destination)
    A_Star(london_graph, source, destination, london_graph.heuristic)

# Run the experiment
experiment5()