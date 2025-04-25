import os
import csv
import numpy as np
from cvrp_solver import CVRPData
from tabu_algorithm import TabuSearchCVRP
from genetic_algorithm import GeneticAlgorithmCVRP

# Configuration
DATA_FOLDER = "data"
OPTIMAL_FOLDER = "data/optimal_data"
INSTANCES = ["A-n32-k5.vrp", "A-n60-k9.vrp"]
RUNS_PER_CONFIG = 10

# Function to read optimal cost from a file
def read_optimal_cost(file_path):
    try:
        with open(file_path, "r") as file:
            for line in file:
                if "Cost" in line:
                    return float(line.split()[-1])
    except FileNotFoundError:
        return "N/A"
    return "N/A"

# Prepare CSV files
os.makedirs("results", exist_ok=True)
TABU_CSV = "results/tabu_tuning_results.csv"
GA_CSV = "results/ga_tuning_results.csv"

# Write headers
with open(TABU_CSV, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Instance", "Optimal", "Configuration", "Best", "Worst", "Average", "Std"])

with open(GA_CSV, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Instance", "Optimal", "Configuration", "Best", "Worst", "Average", "Std"])

# Loop over instances
for instance in INSTANCES:
    file_path = os.path.join(DATA_FOLDER, instance)
    optimal_path = os.path.join(OPTIMAL_FOLDER, instance)
    cvrp_data = CVRPData(file_path)
    optimal_cost = read_optimal_cost(optimal_path)

    print(f"\n\U0001F4E6 Testing Instance: {instance}")

    # ====================
    # TABU SEARCH TUNING
    # ====================
    print("\n\U0001F527 Tuning Tabu Search...\n")
    tabu_configs = [
        {"max_iterations": 200, "tabu_tenure": 5},
        {"max_iterations": 500, "tabu_tenure": 10},
        {"max_iterations": 1000, "tabu_tenure": 15}
    ]

    print(f"{'Optimal':<10}{'Configuration':<60}{'Best':>10}{'Worst':>10}{'Average':>10}{'Std':>10}")
    for config in tabu_configs:
        solver = TabuSearchCVRP(cvrp_data, **config)
        stats = solver.run(runs=RUNS_PER_CONFIG)
        print(f"{str(optimal_cost):<10}{str(config):<60}{stats['best']:>10.2f}{stats['worst']:>10.2f}{stats['avg']:>10.2f}{stats['std']:>10.2f}")

        # Save to CSV
        with open(TABU_CSV, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([instance, optimal_cost, str(config), stats['best'], stats['worst'], stats['avg'], stats['std']])

    # ====================
    # GENETIC ALGORITHM TUNING
    # ====================
    print("\n\U0001F527 Tuning Genetic Algorithm...\n")
    ga_configs = [
        {"population_size": 30, "generations": 200, "crossover_prob": 0.8, "mutation_prob": 0.1,
         "mutation_type": "swap", "crossover_type": "OX"},
        {"population_size": 50, "generations": 500, "crossover_prob": 0.9, "mutation_prob": 0.2,
         "mutation_type": "inversion", "crossover_type": "PMX"},
        {"population_size": 100, "generations": 300, "crossover_prob": 0.7, "mutation_prob": 0.05,
         "mutation_type": "swap", "crossover_type": "PMX"},
        {"population_size": 60, "generations": 400, "crossover_prob": 0.75, "mutation_prob": 0.15,
         "mutation_type": "inversion", "crossover_type": "OX"}
    ]

    print(f"{'Optimal':<10}{'Configuration':<60}{'Best':>10}{'Worst':>10}{'Average':>10}{'Std':>10}")
    for config in ga_configs:
        solver = GeneticAlgorithmCVRP(cvrp_data, **config)
        stats = solver.run(runs=RUNS_PER_CONFIG)
        print(f"{str(optimal_cost):<10}{str(config):<60}{stats['best']:>10.2f}{stats['worst']:>10.2f}{stats['avg']:>10.2f}{stats['std']:>10.2f}")

        # Save to CSV
        with open(GA_CSV, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([instance, optimal_cost, str(config), stats['best'], stats['worst'], stats['avg'], stats['std']])

print("\n✅ Tuning results saved to 'results/' folder!")
