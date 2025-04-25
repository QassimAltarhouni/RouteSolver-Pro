import random
import numpy as np

class GeneticAlgorithmCVRP:
    def __init__(self, cvrp_data, population_size=30, generations=100, crossover_prob=0.7, mutation_prob=0.1  ,
                 mutation_type="swap", crossover_type="OX"):
        self.cvrp = cvrp_data
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.mutation_type = mutation_type  # New
        self.crossover_type = crossover_type  # New

    def evaluate_route(self, route):
        total_distance = 0
        current_capacity = 0
        prev_location = 1

        for customer in route:
            demand = self.cvrp.demands[customer]
            if current_capacity + demand > self.cvrp.capacity:
                total_distance += self.cvrp.distance_matrix[prev_location][1]
                prev_location = 1
                current_capacity = 0

            total_distance += self.cvrp.distance_matrix[prev_location][customer]
            current_capacity += demand
            prev_location = customer

        total_distance += self.cvrp.distance_matrix[prev_location][1]
        return total_distance

    def initialize_population(self):
        customer_ids = list(self.cvrp.locations.keys())[1:]
        population = []
        for _ in range(self.population_size):
            individual = customer_ids.copy()
            random.shuffle(individual)
            population.append(individual)
        return population

    def tournament_selection(self, population, fitnesses, tournament_size=2):
        selected = random.sample(list(zip(population, fitnesses)), tournament_size)
        selected.sort(key=lambda x: x[1])
        return selected[0][0]

    def crossover(self, parent1, parent2):
        if random.random() > self.crossover_prob:
            return parent1.copy()

        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end + 1] = parent1[start:end + 1]

        pointer = 0
        for gene in parent2:
            if gene not in child:
                while child[pointer] is not None:
                    pointer += 1
                child[pointer] = gene
        return child

    def mutate(self, individual):
        if random.random() < self.mutation_prob:
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]

    def run(self, runs=1):
        distances = []

        for _ in range(runs):
            population = self.initialize_population()
            best_individual = min(population, key=self.evaluate_route)
            best_cost = self.evaluate_route(best_individual)

            for _ in range(self.generations):
                fitnesses = [self.evaluate_route(ind) for ind in population]
                new_population = []

                for _ in range(self.population_size):
                    parent1 = self.tournament_selection(population, fitnesses)
                    parent2 = self.tournament_selection(population, fitnesses)
                    child = self.crossover(parent1, parent2)
                    self.mutate(child)
                    new_population.append(child)

                population = new_population
                current_best = min(population, key=self.evaluate_route)
                current_cost = self.evaluate_route(current_best)

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_individual = current_best

            distances.append(best_cost)

        return {
            "best": min(distances),
            "worst": max(distances),
            "avg": np.mean(distances),
            "std": np.std(distances)
        }
