# simulated_annealing.py
import random
import math
import numpy as np

class SimulatedAnnealingCVRP:
    def __init__(self, cvrp_data, initial_temp=1000, cooling_rate=0.995, stopping_temp=1):
        # Step 1: Initialize parameters
        self.cvrp = cvrp_data
        self.temperature = initial_temp  # Starting temperature
        self.cooling_rate = cooling_rate  # Rate of temperature decrease
        self.stopping_temp = stopping_temp  # Minimum temperature to stop

    def evaluate_route(self, route):
        # Step 2: Calculate total travel distance of a route considering capacity
        total_distance = 0
        current_capacity = 0
        prev_location = 1  # Start from depot

        for customer in route:
            demand = self.cvrp.demands[customer]

            if current_capacity + demand > self.cvrp.capacity:
                # Return to depot if capacity exceeded
                total_distance += self.cvrp.distance_matrix[prev_location][1]
                prev_location = 1
                current_capacity = 0

            # Travel to customer
            total_distance += self.cvrp.distance_matrix[prev_location][customer]
            current_capacity += demand
            prev_location = customer

        # Return to depot after last customer
        total_distance += self.cvrp.distance_matrix[prev_location][1]
        return total_distance

    def swap_customers(self, route):
        # Step 3: Generate a neighbor by swapping two customers
        a, b = random.sample(range(len(route)), 2)
        neighbor = route.copy()
        neighbor[a], neighbor[b] = neighbor[b], neighbor[a]
        return neighbor

    def run(self):
        # Step 4: Execute Simulated Annealing process
        customer_ids = list(self.cvrp.locations.keys())[1:]  # Exclude depot
        current_solution = customer_ids.copy()
        random.shuffle(current_solution)  # Initial random solution

        best_solution = current_solution.copy()
        best_cost = self.evaluate_route(best_solution)
        current_cost = best_cost

        while self.temperature > self.stopping_temp:
            # Step 5: Generate and evaluate neighbor
            new_solution = self.swap_customers(current_solution)
            new_cost = self.evaluate_route(new_solution)

            # Step 6: Accept new solution based on probability
            cost_diff = new_cost - current_cost
            if cost_diff < 0 or random.uniform(0, 1) < math.exp(-cost_diff / self.temperature):
                current_solution = new_solution
                current_cost = new_cost

                # Step 7: Update the best solution if better
                if new_cost < best_cost:
                    best_solution = new_solution
                    best_cost = new_cost

            # Step 8: Cool down temperature
            self.temperature *= self.cooling_rate

        # Step 9: Return best found solution and its cost
        return best_solution, best_cost
