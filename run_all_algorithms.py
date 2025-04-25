import os
import csv
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
        generations=500,
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
