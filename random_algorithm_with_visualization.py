# ==============================
# Random Search Algorithm with Statistics and Visualization
# ==============================
import random
import numpy as np  # Import NumPy for statistical calculations
import matplotlib.pyplot as plt  # For visualization
from cvrp_solver import cvrp


class RandomSearchCVRP:
    def __init__(self, cvrp_data, num_iterations=200000):
        """Initialize the Random Search algorithm."""
        self.cvrp = cvrp_data
        self.num_iterations = num_iterations

    def evaluate_route(self, route):
        """Calculates the total travel distance for a given route."""
        total_distance = 0
        current_capacity = 0
        prev_location = 1  # Start at depot
        route_segment = [prev_location]  # Start from depot
        final_routes = []  # Store structured routes

        for customer in route:
            customer_id = customer
            demand = self.cvrp.demands[customer_id]

            if current_capacity + demand > self.cvrp.capacity:
                # Store completed route and return to depot
                final_routes.append(route_segment + [1])
                total_distance += self.cvrp.distance_matrix[prev_location][1]
                prev_location = 1
                current_capacity = 0
                route_segment = [1]  # Start new route from depot

            total_distance += self.cvrp.distance_matrix[prev_location][customer_id]
            route_segment.append(customer_id)
            prev_location = customer_id
            current_capacity += demand

        # Return to depot at the end
        route_segment.append(1)
        final_routes.append(route_segment)
        total_distance += self.cvrp.distance_matrix[prev_location][1]

        return total_distance, final_routes

    def run(self):
        """Runs the Random Search algorithm to find a feasible CVRP solution."""
        best_route = None
        best_distance = float("inf")
        worst_distance = float("-inf")  # Track worst case
        all_distances = []  # Store all distances for analysis
        best_routes_structure = []  # Store best structured routes

        customer_ids = list(self.cvrp.locations.keys())[1:]  # Exclude depot

        for _ in range(self.num_iterations):
            random.shuffle(customer_ids)  # Generate a random route
            distance, routes_structure = self.evaluate_route(customer_ids)
            all_distances.append(distance)  # Store all distances

            if distance < best_distance:
                best_distance = distance
                best_route = customer_ids.copy()
                best_routes_structure = routes_structure  # Store best routes

            if distance > worst_distance:
                worst_distance = distance

        # Compute statistics
        avg_distance = np.mean(all_distances) if all_distances else 0
        std_distance = np.std(all_distances) if all_distances else 0

        return best_route, best_distance, worst_distance, avg_distance, std_distance, best_routes_structure

    def plot_routes(self, routes):
        """Visualizes the best Random Search CVRP routes using Matplotlib."""
        depot_x, depot_y = self.cvrp.locations[1]  # Depot coordinates
        customer_x = [self.cvrp.locations[i][0] for i in self.cvrp.locations if i != 1]
        customer_y = [self.cvrp.locations[i][1] for i in self.cvrp.locations if i != 1]

        plt.figure(figsize=(10, 6))
        plt.scatter(depot_x, depot_y, c='red', marker='s', s=150, label="Depot")
        plt.scatter(customer_x, customer_y, c='blue', marker='o', s=50, label="Customers")

        colors = ['b', 'g', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']

        for i, route in enumerate(routes):
            route_x = [self.cvrp.locations[i][0] for i in route]
            route_y = [self.cvrp.locations[i][1] for i in route]
            plt.plot(route_x, route_y, marker='o', linestyle='-', color=colors[i % len(colors)], label=f"Truck {i+1}")

        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Random Search - CVRP Routes")
        plt.legend()
        plt.grid()
        plt.show()


# Run Random Search
random_search = RandomSearchCVRP(cvrp)
(best_random_route, best_random_distance, worst_random_distance,
 avg_random_distance, std_random_distance, best_routes_structure) = random_search.run()

# Print results
print("\n🎲 Random Search Results:")
print(f"✅ Best Distance: {best_random_distance:.2f}")
print(f"❌ Worst Distance: {worst_random_distance:.2f}")
print(f"📊 Average Distance: {avg_random_distance:.2f}")
print(f"📉 Standard Deviation: {std_random_distance:.2f}")
print(f"🔀 Best Random Route: {best_random_route}")

# Plot the best routes found by Random Search
random_search.plot_routes(best_routes_structure)
