from cvrp_solver import cvrp

class GreedyCVRP:
    """
    Greedy algorithm for CVRP: repeatedly visit the nearest unvisited customer
    that can be served without exceeding vehicle capacity.
    """

    def __init__(self, cvrp_data):
        """
        Initialize the Greedy algorithm with CVRP data.
        :param cvrp_data: An instance of CVRPData.
        """
        self.cvrp = cvrp_data

    def run(self):
        """
        Execute the greedy routing algorithm.
        :return: (routes, total_distance)
                 routes is a list of routes (each a list of node IDs including start/end at 1).
                 total_distance is the total traveled distance.
        """
        unvisited = set(self.cvrp.locations.keys()) - {1}  # Exclude depot (1)
        routes = []
        total_distance = 0.0

        while unvisited:
            route = []
            current_capacity = 0
            current_location = 1  # Start at depot
            route_distance = 0.0

            while True:
                nearest_customer = None
                nearest_distance = float('inf')
                # Find nearest feasible customer
                for customer in unvisited:
                    demand = self.cvrp.demands[customer]
                    if current_capacity + demand <= self.cvrp.capacity:
                        distance = self.cvrp.distance_matrix[current_location, customer]
                        if distance < nearest_distance:
                            nearest_distance = distance
                            nearest_customer = customer

                if nearest_customer is None:
                    # No feasible customer remaining, return to depot
                    route_distance += self.cvrp.distance_matrix[current_location, 1]
                    break

                # Visit the nearest customer
                route.append(nearest_customer)
                current_capacity += self.cvrp.demands[nearest_customer]
                route_distance += nearest_distance
                unvisited.remove(nearest_customer)
                current_location = nearest_customer

                if not unvisited:
                    # All customers visited, return to depot
                    route_distance += self.cvrp.distance_matrix[current_location, 1]
                    break

            routes.append([1] + route + [1])
            total_distance += route_distance

        return routes, total_distance

# Run the Greedy Algorithm
greedy_solver = GreedyCVRP(cvrp)
best_routes, best_distance = greedy_solver.run()

# Print the output
# print("\n🚀 Best Greedy Routes:")
# for i, route in enumerate(best_routes):
#    print(f"Route #{i+1}: {' -> '.join(map(str, route))}")
# print(f"📏 Total Distance: {best_distance:.2f}")