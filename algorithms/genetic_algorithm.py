import random
import numpy as np

class GeneticAlgorithmCVRP:
    """
    Genetic Algorithm for CVRP: evolves a population of routes with crossover and mutation.
    """
    def __init__(self, cvrp_data, population_size=50, generations=100,
                 crossover_prob=0.7, mutation_prob=0.1,
                 mutation_type="swap", crossover_type="OX"):
        """
        Initialize GA parameters.
        :param cvrp_data: An instance of CVRPData.
        :param population_size: Number of individuals.
        :param generations: Number of generations (iterations).
        :param crossover_prob: Probability of crossover.
        :param mutation_prob: Probability of mutation.
        :param mutation_type: Mutation strategy ("swap" or "inversion").
        :param crossover_type: Crossover strategy ("OX", "PMX", etc.).
        """
        self.cvrp = cvrp_data
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.mutation_type = mutation_type
        self.crossover_type = crossover_type

    def evaluate_route(self, route):
        """
        Calculate total distance of a CVRP route.
        """
        total_distance = 0.0
        current_capacity = 0
        prev_location = 1  # start at depot

        for customer in route:
            demand = self.cvrp.demands[customer]
            if current_capacity + demand > self.cvrp.capacity:
                total_distance += self.cvrp.distance_matrix[prev_location][1]
                prev_location = 1
                current_capacity = 0

            total_distance += self.cvrp.distance_matrix[prev_location][customer]
            prev_location = customer
            current_capacity += demand

        total_distance += self.cvrp.distance_matrix[prev_location][1]  # return to depot
        return total_distance

    def initialize_population(self):
        """
        Generate initial random population of routes.
        """
        customer_ids = list(self.cvrp.locations.keys())[1:]  # exclude depot
        population = []
        for _ in range(self.population_size):
            individual = customer_ids.copy()
            random.shuffle(individual)
            population.append(individual)
        return population

    def tournament_selection(self, population, fitnesses, tournament_size=2):
        """
        Select an individual via tournament selection.
        """
        participants = random.sample(list(zip(population, fitnesses)), tournament_size)
        participants.sort(key=lambda x: x[1])  # sort by distance
        return participants[0][0]

    def crossover(self, parent1, parent2):
        """
        Perform Order Crossover (OX).  If no crossover, returns parent copy.
        """
        if random.random() > self.crossover_prob:
            return parent1.copy()

        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end+1] = parent1[start:end+1]
        pointer = 0
        for gene in parent2:
            if gene not in child:
                while child[pointer] is not None:
                    pointer += 1
                child[pointer] = gene
        return child

    def mutate(self, individual):
        """
        Perform swap mutation on an individual route.
        """
        if random.random() < self.mutation_prob:
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]

    def run(self, runs=1):
        """
        Execute the GA and return distance stats.
        :param runs: Number of independent runs.
        :return: Dict with 'best', 'worst', 'avg', 'std' of best distances found.
        """
        best_costs = []
        for _ in range(runs):
            population = self.initialize_population()
            best_individual = min(population, key=self.evaluate_route)
            best_cost = self.evaluate_route(best_individual)

            for _ in range(self.generations):
                fitnesses = [self.evaluate_route(ind) for ind in population]
                new_population = []
                for _ in range(self.population_size):
                    p1 = self.tournament_selection(population, fitnesses)
                    p2 = self.tournament_selection(population, fitnesses)
                    child = self.crossover(p1, p2)
                    self.mutate(child)
                    new_population.append(child)
                population = new_population
                current_best = min(population, key=self.evaluate_route)
                current_cost = self.evaluate_route(current_best)
                if current_cost < best_cost:
                    best_cost = current_cost

            best_costs.append(best_cost)

        arr = np.array(best_costs)
        return {
            "best": float(arr.min()),
            "worst": float(arr.max()),
            "avg": float(arr.mean()),
            "std": float(arr.std())
        }
