import numpy as np
import matplotlib.pyplot as plt
from cvrp_solver import CVRPData
from genetic_algorithm import GeneticAlgorithmCVRP

# Optional: smooth the curve using a moving average
def smooth(values, window=5):
    return np.convolve(values, np.ones(window)/window, mode='valid')

# Load data
cvrp_data = CVRPData("data/A-n60-k9.vrp")

# Configs to compare
configs = [
    {"mutation": "swap", "crossover": "OX"},
    {"mutation": "swap", "crossover": "PMX"},
    {"mutation": "inversion", "crossover": "OX"},
    {"mutation": "inversion", "crossover": "PMX"},
]

generations = 200
population_size = 50

for cfg in configs:
    ga = GeneticAlgorithmCVRP(
        cvrp_data,
        population_size=population_size,
        generations=generations,
        crossover_prob=0.9,
        mutation_prob=0.1,
        mutation_type=cfg["mutation"],
        crossover_type=cfg["crossover"]
    )

    population = ga.initialize_population()

    min_fitness, mean_fitness, max_fitness = [], [], []

    for _ in range(generations):
        fitnesses = [ga.evaluate_route(ind) for ind in population]
        min_fitness.append(min(fitnesses))
        mean_fitness.append(np.mean(fitnesses))
        max_fitness.append(max(fitnesses))

        new_population = []
        for _ in range(population_size):
            p1 = ga.tournament_selection(population, fitnesses)
            p2 = ga.tournament_selection(population, fitnesses)
            child = ga.crossover(p1, p2)
            ga.mutate(child)
            new_population.append(child)
        population = new_population

    # Smooth the data (optional)
    smooth_window = 5
    gens = np.arange(1, generations + 1)
    min_sm = smooth(min_fitness, smooth_window)
    mean_sm = smooth(mean_fitness, smooth_window)
    max_sm = smooth(max_fitness, smooth_window)
    gens_sm = gens[smooth_window - 1:]  # Align with smoothed data

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(gens_sm, min_sm, label="Best (Min)", color="blue", linewidth=2)
    plt.plot(gens_sm, mean_sm, label="Mean", color="green", linewidth=2)
    plt.plot(gens_sm, max_sm, label="Worst (Max)", color="red", linewidth=2)

    plt.title(f"GA Route Cost over Generations\nMutation = {cfg['mutation'].capitalize()}, Crossover = {cfg['crossover']}")
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Route Cost", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"ga_fitness_plot_{cfg['mutation']}_{cfg['crossover']}.png", dpi=300)
    plt.show()
