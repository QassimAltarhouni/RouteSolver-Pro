import random
import math

class SimulatedAnnealingCVRP:
    """
    Simulated Annealing algorithm for CVRP: probabilistically accepts worse
    solutions to escape local minima.
    """
    def __init__(self, cvrp_data, initial_temp=1000.0, cooling_rate=0.995, stopping_temp=1.0):
        """
        Initialize SA parameters.
        :param cvrp_data: An instance of CVRPData.
        :param initial_temp: Starting temperature.
        :param cooling_rate: Factor (0<rate<1) to reduce temperature.
        :param stopping_temp: Temperature threshold to stop.
        """
        self.cvrp = cvrp_data
        self.temperature = initial_temp
        self.cooling_rate = cooling_rate
        self.stopping_temp = stopping_temp

    def evaluate_route(self, route):
        """
        Calculate total distance of a CVRP route, respecting capacity.
        """
        total_distance = 0.0
        current_capacity = 0
        prev_location = 1  # Start at depot

        for customer in route:
            demand = self.cvrp.demands[customer]
            if current_capacity + demand > self.cvrp.capacity:
                # Return to depot if capacity exceeded
                total_distance += self.cvrp.distance_matrix[prev_location][1]
                prev_location = 1
                current_capacity = 0
            total_distance += self.cvrp.distance_matrix[prev_location][customer]
            current_capacity += demand
            prev_location = customer

        total_distance += self.cvrp.distance_matrix[prev_location][1]  # return to depot
        return total_distance

    def swap_customers(self, route):
        """
        Generate a neighbor by swapping two customers in the route.
        """
        a, b = random.sample(range(len(route)), 2)
        neighbor = route.copy()
        neighbor[a], neighbor[b] = neighbor[b], neighbor[a]
        return neighbor

    def run(self):
        """
        Execute the Simulated Annealing process and return best route and cost.
        :return: Dict with 'best_route' and 'best_cost'.
        """
        current_solution = list(self.cvrp.locations.keys())[1:]  # exclude depot
        random.shuffle(current_solution)
        best_solution = current_solution.copy()
        best_cost = self.evaluate_route(best_solution)
        current_cost = best_cost

        while self.temperature > self.stopping_temp:
            new_solution = self.swap_customers(current_solution)
            new_cost = self.evaluate_route(new_solution)
            cost_diff = new_cost - current_cost

            # Accept new solution by Metropolis criterion
            if cost_diff < 0 or random.uniform(0, 1) < math.exp(-cost_diff / self.temperature):
                current_solution = new_solution
                current_cost = new_cost
                if new_cost < best_cost:
                    best_solution = new_solution.copy()
                    best_cost = new_cost

            # Cool down
            self.temperature *= self.cooling_rate

        return {"best_route": best_solution, "best_cost": best_cost}
