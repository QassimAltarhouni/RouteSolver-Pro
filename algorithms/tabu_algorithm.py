import random
import numpy as np
from collections import deque
import heapq

class TabuSearchCVRP:
    """
    Tabu Search algorithm for CVRP: improves routes using a tabu list to escape local minima.
    """
    def __init__(self, cvrp_data, tabu_tenure=10, max_iterations=5000, neighbor_sample_size=100):
        self.cvrp = cvrp_data
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.neighbor_sample_size = neighbor_sample_size

    def evaluate_route(self, route): # while cnt < max_fitness_evals
        total_distance = 0.0
        current_capacity = 0
        prev_location = 1  # Start at depot

        for customer in route:
            demand = self.cvrp.demands[customer]
            if current_capacity + demand > self.cvrp.capacity:
                total_distance += self.cvrp.distance_matrix[prev_location][1]
                prev_location = 1
                current_capacity = 0

            total_distance += self.cvrp.distance_matrix[prev_location][customer]
            current_capacity += demand
            prev_location = customer

        total_distance += self.cvrp.distance_matrix[prev_location][1]
        return total_distance

    def generate_neighbors(self, route):
        neighbors = []
        n = len(route)
        for i in range(n):
            for j in range(i + 1, n):
                neighbor = route[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append((i, j, neighbor))
        return neighbors

    def run(self, runs=1):
        global best_solution
        best_costs = []
        customer_ids = list(self.cvrp.locations.keys())[1:]  # exclude depot
        sample_counter = 0
        for _ in range(runs):
            current_solution = customer_ids.copy()
            random.shuffle(current_solution)
            best_solution = current_solution
            best_cost = self.evaluate_route(best_solution)

            tabu_queue = deque()
            tabu_set = set()

            for _ in range(self.max_iterations):
                neighbors = self.generate_neighbors(current_solution)
                # Sample a limited number of neighbors
                sampled_neighbors = random.sample(
                    neighbors,
                    min(self.neighbor_sample_size, len(neighbors))
                )

                # Evaluate and get the best one using heapq (faster than sort)
                neighbor_evals = [
                    (i, j, neighbor, self.evaluate_route(neighbor))

                    for i, j, neighbor in sampled_neighbors
                ]
                sample_counter += len(sampled_neighbors)
                top_neighbors = heapq.nsmallest(1, neighbor_evals, key=lambda x: x[3])

                if not top_neighbors:
                    continue

                i, j, neighbor, cost = top_neighbors[0]
                move = (i, j)

                if move not in tabu_set or cost < best_cost:
                    current_solution = neighbor
                    if cost < best_cost:
                        best_solution = neighbor
                        best_cost = cost
                    tabu_queue.append(move)
                    tabu_set.add(move)
                    if len(tabu_queue) > self.tabu_tenure:
                        old_move = tabu_queue.popleft()
                        tabu_set.discard(old_move)

            best_costs.append(best_cost)

        arr = np.array(best_costs)
        print(f"Total samples evaluated: {sample_counter}")
        return {
            "best": float(arr.min()),
            "worst": float(arr.max()),
            "avg": float(arr.mean()),
            "std": float(arr.std()),
            "route": best_solution
        }
