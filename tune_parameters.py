# tune_parameters.py

import os
import numpy as np
from cvrp_solver import CVRPData
from tabu_algorithm import TabuSearchCVRP
from simulated_annealing import SimulatedAnnealingCVRP
from genetic_algorithm import GeneticAlgorithmCVRP

# Configuration
DATA_FOLDER = "data"
INSTANCES = ["A-n32-k5.vrp", "A-n60-k9.vrp"]  # Easy and hard
RUNS_PER_CONFIG = 10

# Summarize stats
def summarize(distances):
    return {
        "best": min(distances),
        "worst": max(distances),
        "avg": np.mean(distances),
        "std": np.std(distances)
    }

# Loop over both easy and hard instance
for instance in INSTANCES:
    file_path = os.path.join(DATA_FOLDER, instance)
    cvrp_data = CVRPData(file_path)

    print(f"\n📦 Testing Instance: {instance}")

    # ====================
    # TABU SEARCH TUNING
    # ====================
    print("\n🔧 Tuning Tabu Search...\n")
    tabu_configs = [
        {"max_iterations": 200, "tabu_tenure": 5},
        {"max_iterations": 500, "tabu_tenure": 10},
        {"max_iterations": 1000, "tabu_tenure": 15}
    ]

    for config in tabu_configs:
        results = []
        for _ in range(RUNS_PER_CONFIG):
            solver = TabuSearchCVRP(cvrp_data, **config)
            _, dist = solver.run()
            results.append(dist)

        stats = summarize(results)
        print(f"⚙️ Tabu Config {config} → Best: {stats['best']:.2f}, Worst: {stats['worst']:.2f}, "
              f"Avg: {stats['avg']:.2f}, Std: {stats['std']:.2f}")

    # ==============================
    # SIMULATED ANNEALING TUNING
    # ==============================
    print("\n🔧 Tuning Simulated Annealing...\n")
    sa_configs = [
        {"initial_temp": 1000, "cooling_rate": 0.995, "stopping_temp": 10},
        {"initial_temp": 500, "cooling_rate": 0.99, "stopping_temp": 1},
        {"initial_temp": 2000, "cooling_rate": 0.98, "stopping_temp": 5}
    ]

    for config in sa_configs:
        results = []
        for _ in range(RUNS_PER_CONFIG):
            solver = SimulatedAnnealingCVRP(cvrp_data, **config)
            _, dist = solver.run()
            results.append(dist)

        stats = summarize(results)
        print(f"⚙️ SA Config {config} → Best: {stats['best']:.2f}, Worst: {stats['worst']:.2f}, "
              f"Avg: {stats['avg']:.2f}, Std: {stats['std']:.2f}")

    # ==============================
    # GENETIC ALGORITHM TUNING
    # ==============================
    print("\n🔧 Tuning Genetic Algorithm...\n")
    ga_configs = [
        {"population_size": 30, "generations": 200, "crossover_prob": 0.8, "mutation_prob": 0.1, "mutation_type": "swap", "crossover_type": "OX"},
        {"population_size": 50, "generations": 500, "crossover_prob": 0.9, "mutation_prob": 0.2, "mutation_type": "inversion", "crossover_type": "PMX"},
        {"population_size": 100, "generations": 300, "crossover_prob": 0.7, "mutation_prob": 0.05, "mutation_type": "swap", "crossover_type": "PMX"},
        {"population_size": 60, "generations": 400, "crossover_prob": 0.75, "mutation_prob": 0.15, "mutation_type": "inversion", "crossover_type": "OX"}
    ]

    for config in ga_configs:
        results = []
        for _ in range(RUNS_PER_CONFIG):
            solver = GeneticAlgorithmCVRP(cvrp_data, **config)
            _, dist = solver.run()
            results.append(dist)

        stats = summarize(results)
        print(f"⚙️ GA Config {config} → Best: {stats['best']:.2f}, Worst: {stats['worst']:.2f}, "
              f"Avg: {stats['avg']:.2f}, Std: {stats['std']:.2f}")
