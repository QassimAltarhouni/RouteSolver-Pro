# ==============================
# Random Search Algorithm with Statistics
# ==============================
import random
import numpy as np  # Import NumPy for statistical calculations
from cvrp_solver import cvrp
import json  # Import JSON module


class RandomSearchCVRP:
    def __init__(self, cvrp_data, num_iterations=1000):
        """Initialize the Random Search algorithm."""
        self.cvrp = cvrp_data
        self.num_iterations = num_iterations

    def evaluate_route(self, route):
        """Calculates the total travel distance for a given route."""
        total_distance = 0
        current_capacity = 0
        prev_location = 1  # Start at depot

        for customer in route:
            customer_id = customer
            demand = self.cvrp.demands[customer_id]

            if current_capacity + demand > self.cvrp.capacity:
                # Return to depot before continuing
                total_distance += self.cvrp.distance_matrix[prev_location][1]
                prev_location = 1
                current_capacity = 0

            total_distance += self.cvrp.distance_matrix[prev_location][customer_id]
            prev_location = customer_id
            current_capacity += demand

        # Return to depot at the end
        total_distance += self.cvrp.distance_matrix[prev_location][1]
        return total_distance

    def run(self):
        """Runs the Random Search algorithm to find a feasible CVRP solution."""
        best_route = None
        best_distance = float("inf")
        worst_distance = float("-inf")  # Track worst case
        all_distances = []  # Store all distances for analysis

        customer_ids = list(self.cvrp.locations.keys())[1:]  # Exclude depot

        for _ in range(self.num_iterations):
            random.shuffle(customer_ids)  # Generate a random route
            distance = self.evaluate_route(customer_ids)
            all_distances.append(distance)  # Store all distances

            if distance < best_distance:
                best_distance = distance
                best_route = customer_ids.copy()

            if distance > worst_distance:
                worst_distance = distance

        # Compute statistics
        avg_distance = np.mean(all_distances) if all_distances else 0
        std_distance = np.std(all_distances) if all_distances else 0

        return best_route, best_distance, worst_distance, avg_distance, std_distance


# Run Random Search
random_search = RandomSearchCVRP(cvrp)
best_random_route, best_random_distance, worst_random_distance, avg_random_distance, std_random_distance = random_search.run()

# Print results
# print("\n🎲 Random Search Results:")
# print(f"✅ Best Distance: {best_random_distance:.2f}")
#print(f"❌ Worst Distance: {worst_random_distance:.2f}")
#print(f"📊 Average Distance: {avg_random_distance:.2f}")
#print(f"📉 Standard Deviation: {std_random_distance:.2f}")
#print(f"🔀 Best Random Route: {best_random_route}")




