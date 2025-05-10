import random
import numpy as np

class RandomSearchCVRP:
    """
    Random Search algorithm for CVRP: generates random routes and reports statistics.
    """
    def __init__(self, cvrp_data, max_fitness_evals=5000):
        self.cvrp = cvrp_data
        self.max_fitness_evals = max_fitness_evals

    def evaluate_route(self, route):
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
            prev_location = customer
            current_capacity += demand

        total_distance += self.cvrp.distance_matrix[prev_location][1]
        return total_distance

    def split_into_routes(self, flat_route):
        routes = []
        route = [1]
        current_capacity = 0

        for customer in flat_route:
            demand = self.cvrp.demands[customer]
            if current_capacity + demand > self.cvrp.capacity:
                route.append(1)
                routes.append(route)
                route = [1, customer]
                current_capacity = demand
            else:
                route.append(customer)
                current_capacity += demand

        route.append(1)
        routes.append(route)
        return routes

    def run_multiple(self, runs=10):
        best_costs = []
        best_overall_route = None

        customer_ids = list(self.cvrp.locations.keys())[1:]

        for _ in range(runs):
            best_cost = float("inf")
            best_route = None

            for _ in range(self.max_fitness_evals):
                route = customer_ids.copy()
                random.shuffle(route)
                dist = self.evaluate_route(route)

                if dist < best_cost:
                    best_cost = dist
                    best_route = route.copy()

            best_costs.append(best_cost)
            if best_overall_route is None or best_cost < self.evaluate_route(best_overall_route):
                best_overall_route = best_route

        arr = np.array(best_costs)
        return {
            "best": float(arr.min()),
            "worst": float(arr.max()),
            "avg": float(arr.mean()),
            "std": float(arr.std()),
            "route": best_overall_route,
            "split_routes": self.split_into_routes(best_overall_route)
        }
