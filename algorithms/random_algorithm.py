import random
import numpy as np

from cvrp_solver import cvrp


class RandomSearchCVRP:
    """
    Random Search algorithm for CVRP: generates random routes and reports statistics.
    """
    def __init__(self, cvrp_data, max_fitness_evals=5000):
        """
        Initialize the Random Search algorithm.
        :param cvrp_data: An instance of CVRPData.
        :param num_iterations: Number of random routes to generate.
        """
        self.cvrp = cvrp_data
        self.max_fitness_evals = max_fitness_evals

    def evaluate_route(self, route):
        """
        Calculate total distance of a given route sequence.
        :param route: List of customer node IDs (in visit order).
        """
        total_distance = 0.0
        current_capacity = 0
        prev_location = 1  # Start at depot

        for customer in route:
            demand = self.cvrp.demands[customer]
            if current_capacity + demand > self.cvrp.capacity:
                # Return to depot due to capacity limit
                total_distance += self.cvrp.distance_matrix[prev_location][1]
                prev_location = 1
                current_capacity = 0

            total_distance += self.cvrp.distance_matrix[prev_location][customer]
            prev_location = customer
            current_capacity += demand

        # Return to depot after last customer
        total_distance += self.cvrp.distance_matrix[prev_location][1]
        return total_distance

    def run_multiple(self, runs=10):
        """
        Run the random search multiple times and return aggregated stats.
        """
        best_costs = []
        for _ in range(runs):
            distances = []
            customer_ids = list(self.cvrp.locations.keys())[1:]
            for _ in range(self.max_fitness_evals):
                route = customer_ids.copy()
                random.shuffle(route)
                dist = self.evaluate_route(route)
                distances.append(dist)
            best_costs.append(min(distances))

        arr = np.array(best_costs)
        return {
            "best": float(arr.min()),
            "worst": float(arr.max()),
            "avg": float(arr.mean()),
            "std": float(arr.std())
        }


# Run Random Search

# Print results
# print("\n🎲 Random Search Results:")
# print(f"✅ Best Distance: {best_random_distance:.2f}")
#print(f"❌ Worst Distance: {worst_random_distance:.2f}")
#print(f"📊 Average Distance: {avg_random_distance:.2f}")
#print(f"📉 Standard Deviation: {std_random_distance:.2f}")
#print(f"🔀 Best Random Route: {best_random_route}")




