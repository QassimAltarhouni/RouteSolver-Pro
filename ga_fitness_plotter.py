import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from cvrp_solver import CVRPData
from greedy_algorithm import GreedyCVRP
from random_algorithm import RandomSearchCVRP
from tabu_algorithm import TabuSearchCVRP
from genetic_algorithm import GeneticAlgorithmCVRP

# Paths to data files
DATA_FOLDER = "data"
OPTIMAL_FOLDER = "data/optimal_data"

# List all VRP files in the data directory
vrp_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".vrp")]

# Store results for printing
results = []

# Prepare CSV file
os.makedirs("results", exist_ok=True)
COMPARISON_CSV = "results/algorithm_comparison_results.csv"

# Write header
with open(COMPARISON_CSV, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "File Name", "Optimal", "Greedy",
        "Rand Best", "Rand Worst", "Rand Avg", "Rand Std",
        "Tabu Best", "Tabu Worst", "Tabu Avg", "Tabu Std",
        "GA Best", "GA Worst", "GA Avg", "GA Std"
    ])

# Function to read optimal cost from a file
def read_optimal_cost(file_path):
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                if "Cost" in line:
                    return float(line.split()[-1])
    except FileNotFoundError:
        return None
    return None

# Optional: smooth the curve using a moving average
def smooth(values, window=5):
    return np.convolve(values, np.ones(window)/window, mode='valid')

# Process each VRP file
for file_name in vrp_files:
    file_path = os.path.join(DATA_FOLDER, file_name)
    optimal_file_path = os.path.join(OPTIMAL_FOLDER, file_name)

    cvrp_data = CVRPData(file_path)

    # Greedy (single run only)
    greedy_solver = GreedyCVRP(cvrp_data)
    _, greedy_distance = greedy_solver.run()

    # Random Search (already includes stats)
    random_solver = RandomSearchCVRP(cvrp_data, num_iterations=100)
    _, best_rd, worst_rd, avg_rd, std_rd = random_solver.run()

    # Tabu Search (returns all stats)
    tabu_solver = TabuSearchCVRP(cvrp_data, max_iterations=100)
    tabu_stats = tabu_solver.run(runs=10)

    # Genetic Algorithm (returns all stats)
    ga_solver = GeneticAlgorithmCVRP(
        cvrp_data,
        population_size=50,
        generations=10,
        crossover_prob=0.8,
        mutation_prob=0.1
    )
    ga_stats = ga_solver.run(runs=100)

    # Read optimal cost
    optimal_cost = read_optimal_cost(optimal_file_path)

    # Store all results
    results.append({
        "file": file_name,
        "optimal_cost": optimal_cost if optimal_cost else "N/A",
        "greedy_distance": greedy_distance,

        "best_random_distance": best_rd,
        "worst_random_distance": worst_rd,
        "avg_random_distance": avg_rd,
        "std_random_distance": std_rd,

        "tabu_best": tabu_stats["best"],
        "tabu_worst": tabu_stats["worst"],
        "tabu_avg": tabu_stats["avg"],
        "tabu_std": tabu_stats["std"],

        "ga_best": ga_stats["best"],
        "ga_worst": ga_stats["worst"],
        "ga_avg": ga_stats["avg"],
        "ga_std": ga_stats["std"]
    })

# Print results
print("\n========== FINAL RESULTS ==========")
print(f"{'File Name':<15}{'Optimal':<10}{'Greedy':<10}"
      f"{'Rand Best':<12}{'Rand Worst':<12}{'Rand Avg':<12}{'Rand Std':<12}"
      f"{'Tabu Best':<12}{'Tabu Worst':<12}{'Tabu Avg':<12}{'Tabu Std':<12}"
      f"{'GA Best':<12}{'GA Worst':<12}{'GA Avg':<12}{'GA Std':<12}")
print("-" * 180)

# Save results to CSV
with open(COMPARISON_CSV, mode="a", newline="") as file:
    writer = csv.writer(file)

    for r in results:
        print(f"{r['file']:<15}{r['optimal_cost']:<10}{r['greedy_distance']:<10.2f}"
              f"{r['best_random_distance']:<12.2f}{r['worst_random_distance']:<12.2f}{r['avg_random_distance']:<12.2f}{r['std_random_distance']:<12.2f}"
              f"{r['tabu_best']:<12.2f}{r['tabu_worst']:<12.2f}{r['tabu_avg']:<12.2f}{r['tabu_std']:<12.2f}"
              f"{r['ga_best']:<12.2f}{r['ga_worst']:<12.2f}{r['ga_avg']:<12.2f}{r['ga_std']:<12.2f}")

        writer.writerow([
            r['file'], r['optimal_cost'], r['greedy_distance'],
            r['best_random_distance'], r['worst_random_distance'], r['avg_random_distance'], r['std_random_distance'],
            r['tabu_best'], r['tabu_worst'], r['tabu_avg'], r['tabu_std'],
            r['ga_best'], r['ga_worst'], r['ga_avg'], r['ga_std']
        ])

print("\n✅ Comparison results saved to 'results/' folder!")

# Advanced plot for GA results (optional)
cvrp_data = CVRPData("data/A-n60-k9.vrp")
optimal_cost = 1140  # Replace with real value

generations = 500
population_size = 50
configs = [
    {"mutation": "swap", "crossover": "OX"},
    {"mutation": "swap", "crossover": "PMX"},
    {"mutation": "inversion", "crossover": "OX"},
    {"mutation": "inversion", "crossover": "PMX"},
]

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
    min_fitness, mean_fitness = [], []

    for _ in range(generations):
        fitnesses = [ga.evaluate_route(ind) for ind in population]
        min_fitness.append(min(fitnesses))
        mean_fitness.append(np.mean(fitnesses))

        new_population = []
        for _ in range(population_size):
            p1 = ga.tournament_selection(population, fitnesses)
            p2 = ga.tournament_selection(population, fitnesses)
            child = ga.crossover(p1, p2)
            ga.mutate(child)
            new_population.append(child)
        population = new_population

    # Smooth the data
    smooth_window = 5
    gens = np.arange(1, generations + 1)
    min_sm = smooth(min_fitness, smooth_window)
    mean_sm = smooth(mean_fitness, smooth_window)
    gens_sm = gens[smooth_window - 1:]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(gens_sm, min_sm, label="Best (Min)", color="blue", linewidth=2)
    plt.plot(gens_sm, mean_sm, label="Mean", color="green", linewidth=2, alpha=0.6)
    plt.axhline(y=optimal_cost, color='black', linestyle='--', label='Optimal Cost')

    plt.title(f"GA Route Cost over Generations\nMutation = {cfg['mutation'].capitalize()}, Crossover = {cfg['crossover']}")
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Route Cost", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"results/ga_fitness_plot_{cfg['mutation']}_{cfg['crossover']}.png", dpi=300)
    plt.show()
