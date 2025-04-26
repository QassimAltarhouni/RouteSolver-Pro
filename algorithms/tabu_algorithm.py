import random
import numpy as np

class TabuSearchCVRP:
    """
    Tabu Search algorithm for CVRP: improves routes using a tabu list to escape local minima.
    """
    def __init__(self, cvrp_data, tabu_tenure=10, max_iterations=100):
        """
        Initialize Tabu Search parameters.
        :param cvrp_data: An instance of CVRPData.
        :param tabu_tenure: Number of moves to keep in tabu list.
        :param max_iterations: Max number of iterations per run.
        """
        self.cvrp = cvrp_data
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations

    def evaluate_route(self, route):
        """
        Calculate total distance of a CVRP route.
        :param route: List of customer node IDs.
        """
        total_distance = 0.0
        current_capacity = 0
        prev_location = 1  # Start at depot

        for customer in route:
            demand = self.cvrp.demands[customer]
            if current_capacity + demand > self.cvrp.capacity:
                # Return to depot before continuing
                total_distance += self.cvrp.distance_matrix[prev_location][1]
                prev_location = 1
                current_capacity = 0

            total_distance += self.cvrp.distance_matrix[prev_location][customer]
            current_capacity += demand
            prev_location = customer

        # Return to depot after last customer
        total_distance += self.cvrp.distance_matrix[prev_location][1]
        return total_distance

    def generate_neighbors(self, route):
        """
        Generate neighbor solutions by swapping every pair of customers.
        :param route: Current route (list of customers).
        :return: List of (i, j, neighbor_route) tuples.
        """
        neighbors = []
        n = len(route)
        for i in range(n):
            for j in range(i + 1, n):
                neighbor = route.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append((i, j, neighbor))
        return neighbors

    def run(self, runs=1):
        """
        Execute Tabu Search and return distance stats.
        :param runs: Number of independent runs.
        :return: Dict with 'best', 'worst', 'avg', 'std' distances.
        """
        best_costs = []
        customer_ids = list(self.cvrp.locations.keys())[1:]  # exclude depot

        for _ in range(runs):
            current_solution = customer_ids.copy()
            random.shuffle(current_solution)
            best_solution = current_solution.copy()
            best_cost = self.evaluate_route(best_solution)
            tabu_list = []

            for _ in range(self.max_iterations):
                neighbors = self.generate_neighbors(current_solution)
                # sort neighbors by distance (ascending)
                neighbors.sort(key=lambda x: self.evaluate_route(x[2]))
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

            best_costs.append(best_cost)

        arr = np.array(best_costs)
        return {
            "best": float(arr.min()),
            "worst": float(arr.max()),
            "avg": float(arr.mean()),
            "std": float(arr.std())
        }

