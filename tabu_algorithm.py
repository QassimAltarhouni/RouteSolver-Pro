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

    def run(self):
        # Step 4: Execute the Tabu Search

        # Initialize solution with random shuffle of customers (excluding depot)
        customer_ids = list(self.cvrp.locations.keys())[1:]
        current_solution = customer_ids.copy()
        random.shuffle(current_solution)

        # Set best solution as the initial one
        best_solution = current_solution.copy()
        best_cost = self.evaluate_route(best_solution)

        tabu_list = []  # Initialize empty tabu list

        for iteration in range(self.max_iterations):
            # Step 5: Generate and sort neighbors by cost (lowest first)
            neighbors = self.generate_neighbors(current_solution)
            neighbors = sorted(neighbors, key=lambda x: self.evaluate_route(x[2]))

            for i, j, neighbor in neighbors:
                move = (i, j)  # This is the swap move
                cost = self.evaluate_route(neighbor)

                # Step 6: Choose best allowed move (not in tabu or better than current best)
                if move not in tabu_list or cost < best_cost:
                    current_solution = neighbor

                    # Step 7: Update best solution if found
                    if cost < best_cost:
                        best_solution = neighbor
                        best_cost = cost

                    # Step 8: Add move to tabu list
                    tabu_list.append(move)
                    if len(tabu_list) > self.tabu_tenure:
                        tabu_list.pop(0)  # Maintain tabu list size
                    break  # Apply only one move per iteration

        # Step 9: Return best found solution and its cost
        return best_solution, best_cost
