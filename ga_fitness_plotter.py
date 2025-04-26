import numpy as np
import matplotlib.pyplot as plt

from algorithms.genetic_algorithm import GeneticAlgorithmCVRP
from cvrp_solver import CVRPData


def smooth(values, window=5):
    """
    Simple moving average smoothing function.
    """
    return np.convolve(values, np.ones(window) / window, mode='valid')

def main():
    # Load CVRP data
    cvrp_data = CVRPData("data/A-n32-k5.vrp")
    optimal_cost = 784  # known optimal for this instance

    configs = [
        {"mutation_type": "swap", "crossover_type": "OX"},
        {"mutation_type": "swap", "crossover_type": "PMX"},
        {"mutation_type": "inversion", "crossover_type": "OX"},
        {"mutation_type": "inversion", "crossover_type": "PMX"}
    ]
    population_size = 100
    generations = 500

    for cfg in configs:
        ga = GeneticAlgorithmCVRP(
            cvrp_data,
            population_size=population_size,
            generations=generations,
            crossover_prob=0.8,
            mutation_prob=0.1,
            mutation_type=cfg["mutation_type"],
            crossover_type=cfg["crossover_type"]
        )

        # Evolve GA population and record stats
        population = ga.initialize_population()
        min_fitness, max_fitness, mean_fitness, std_fitness = [], [], [], []
        for _ in range(generations):
            fitnesses = [ga.evaluate_route(ind) for ind in population]
            min_fitness.append(min(fitnesses))
            max_fitness.append(max(fitnesses))
            mean_fitness.append(np.mean(fitnesses))
            std_fitness.append(np.std(fitnesses))
            # Evolve population
            new_population = []
            for _ in range(population_size):
                p1 = ga.tournament_selection(population, fitnesses)
                p2 = ga.tournament_selection(population, fitnesses)
                child = ga.crossover(p1, p2)
                ga.mutate(child)
                new_population.append(child)
            population = new_population

        # Smooth and plot fitness curves
        window = 5
        gens = np.arange(1, generations + 1)
        gens_sm = gens[window - 1:]
        min_sm = smooth(min_fitness, window)
        max_sm = smooth(max_fitness, window)
        mean_sm = smooth(mean_fitness, window)

        plt.figure(figsize=(10, 6))
        plt.plot(gens_sm, min_sm, label="Best (Min)", color="blue")
        plt.plot(gens_sm, max_sm, label="Worst (Max)", color="red")
        plt.plot(gens_sm, mean_sm, label="Mean", color="green")
        plt.axhline(y=optimal_cost, color='black', linestyle='--', label='Optimal Cost')
        plt.title(f"GA Fitness (mutation={cfg['mutation_type']}, crossover={cfg['crossover_type']})")
        plt.xlabel("Generation")
        plt.ylabel("Route Cost")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/ga_fitness_{cfg['mutation_type']}_{cfg['crossover_type']}.png", dpi=300)
        plt.show()

        print(f"Config: mutation={cfg['mutation_type']}, crossover={cfg['crossover_type']}")
        print(f"Best: {min_fitness[-1]:.2f}, Worst: {max_fitness[-1]:.2f}, "
              f"Avg: {mean_fitness[-1]:.2f}")

if __name__ == "__main__":
    main()
