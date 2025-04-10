 # ==============================
# Greedy Algorithm Class
# ==============================
from cvrp_solver import cvrp


class GreedyCVRP:
    def __init__(self, cvrp_data):
        """Initialize the Greedy Algorithm with CVRP data."""
        self.cvrp = cvrp_data




    def run(self):
        """Runs the Greedy Algorithm to construct routes for CVRP."""
        unvisited = set(self.cvrp.locations.keys()) - {1}  # Exclude depot (ID 1)

        routes = []  # List to store all routes
        total_distance = 0  # Total traveled distance

        while unvisited:
            route = []  # Store the current route
            current_capacity = 0  # Keep track of vehicle load
            current_location = 1  # Start at the depot
            route_distance = 0  # Distance traveled in the current route

            while unvisited:
                # Find the nearest customer that fits in the truck
                nearest_customer = None
                nearest_distance = float("inf")

                for customer in unvisited:
                    demand = self.cvrp.demands[customer]

                    if current_capacity + demand <= self.cvrp.capacity:
                        distance = self.cvrp.distance_matrix[current_location, customer]
                        if distance < nearest_distance:
                            nearest_customer = customer
                            nearest_distance = distance

                if nearest_customer is None:
                    # No more customers can be visited, return to depot
                    route_distance += self.cvrp.distance_matrix[current_location, 1]
                    total_distance += route_distance
                    route.append(1)  # Return to depot
                    break

                # Move to the nearest customer
                route.append(nearest_customer)
                current_capacity += self.cvrp.demands[nearest_customer]
                route_distance += nearest_distance
                unvisited.remove(nearest_customer)
                current_location = nearest_customer

            # Return to depot at the end of the route
            route.append(1)
            total_distance += self.cvrp.distance_matrix[current_location, 1]
            routes.append(route)

        return routes, total_distance
# Run the Greedy Algorithm
greedy_solver = GreedyCVRP(cvrp)
best_routes, best_distance = greedy_solver.run()

# Print the output
# print("\n🚀 Best Greedy Routes:")
# for i, route in enumerate(best_routes):
#    print(f"Route #{i+1}: {' -> '.join(map(str, route))}")
# print(f"📏 Total Distance: {best_distance:.2f}")