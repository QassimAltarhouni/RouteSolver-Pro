# tabu_algorithm.py
import random
import numpy as np

class TabuSearchCVRP:
    def __init__(self, cvrp_data, tabu_tenure=10, max_iterations=100):
        # Step 1: Initialize Tabu Search parameters
        self.cvrp = cvrp_data  # CVRP problem instance
        num_customers = len(cvrp_data.locations) - 1  # Exclude depot

        # Dynamically determine tabu_tenure if not provided
        self.tabu_tenure = tabu_tenure if tabu_tenure is not None else max(5, int(0.01 * num_customers))


        self.max_iterations = max_iterations  # Number of search iterations
        self.tabu_list = []  # List of forbidden moves

    def evaluate_route(self, route):
        # Step 2: Calculate total travel distance of a route
        total_distance = 0
        current_capacity = 0
        prev_location = 1  # Start at depot (ID 1)

        for customer in route:
            demand = self.cvrp.demands[customer]

            if current_capacity + demand > self.cvrp.capacity:
                # If adding this customer exceeds capacity, return to depot
                total_distance += self.cvrp.distance_matrix[prev_location][1]
                prev_location = 1
                current_capacity = 0

            # Move to customer and accumulate distance and demand
            total_distance += self.cvrp.distance_matrix[prev_location][customer]
            current_capacity += demand
            prev_location = customer

        # After last customer, return to depot
        total_distance += self.cvrp.distance_matrix[prev_location][1]
        return total_distance

    def generate_neighbors(self, route):
        # Step 3: Generate neighbor solutions by swapping two customer positions
        neighbors = []
        n = len(route)
        for i in range(n):
            for j in range(i + 1, n):
                neighbor = route.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]  # Swap
                neighbors.append((i, j, neighbor))
        return neighbors

    def run(self, runs=1):
        distances = []

        for _ in range(runs):
            customer_ids = list(self.cvrp.locations.keys())[1:]
            current_solution = customer_ids.copy()
            random.shuffle(current_solution)

            best_solution = current_solution.copy()
            best_cost = self.evaluate_route(best_solution)
            tabu_list = []

            for _ in range(self.max_iterations):
                neighbors = self.generate_neighbors(current_solution)
                neighbors = sorted(neighbors, key=lambda x: self.evaluate_route(x[2]))

                for i, j, neighbor in neighbors:
                    move = (i, j)
                    cost = self.evaluate_route(neighbor)
                    if move not in tabu_list or cost < best_cost:
                        current_solution = neighbor
                        if cost < best_cost:
                            best_solution = neighbor
                            best_cost = cost
                        tabu_list.append(move)
                        if len(tabu_list) > self.tabu_tenure:
                            tabu_list.pop(0)
                        break

            distances.append(best_cost)

        return {
            "best": min(distances),
            "worst": max(distances),
            "avg": np.mean(distances),
            "std": np.std(distances)
        }

