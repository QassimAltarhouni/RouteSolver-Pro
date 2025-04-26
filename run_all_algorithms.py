import os
import csv

from algorithms.genetic_algorithm import GeneticAlgorithmCVRP
from algorithms.greedy_algorithm import GreedyCVRP
from algorithms.random_algorithm import RandomSearchCVRP
from algorithms.tabu_algorithm import TabuSearchCVRP
from cvrp_solver import CVRPData


def read_optimal_cost(file_path):
    """
    Reads the optimal cost from a solution file (if exists).
    """
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if "Cost" in line:
                    return float(line.split()[-1])
    except FileNotFoundError:
        return None
    return None

def main():
    DATA_FOLDER = "data"
    OPTIMAL_FOLDER = "data/optimal_data"
    RESULTS_CSV = "results/algorithm_comparison_results.csv"

    os.makedirs("results", exist_ok=True)
    vrp_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".vrp")]

    # Prepare CSV output
    with open(RESULTS_CSV, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "File Name", "Optimal", "Greedy",
            "Rand Best", "Rand Worst", "Rand Avg", "Rand Std",
            "Tabu Best", "Tabu Worst", "Tabu Avg", "Tabu Std",
            "GA Best", "GA Worst", "GA Avg", "GA Std"
        ])

    results = []
    for file_name in vrp_files:
        file_path = os.path.join(DATA_FOLDER, file_name)
        optimal_file = os.path.join(OPTIMAL_FOLDER, file_name)
        cvrp_data = CVRPData(file_path)
        optimal_cost = read_optimal_cost(optimal_file)

        # Greedy (single run)
        greedy_solver = GreedyCVRP(cvrp_data)
        _, greedy_distance = greedy_solver.run()

        # Random Search
        random_solver = RandomSearchCVRP(cvrp_data, num_iterations=100)
        rand_stats = random_solver.run()

        # Tabu Search
        tabu_solver = TabuSearchCVRP(cvrp_data, max_iterations=100)
        tabu_stats = tabu_solver.run(runs=10)

        # Genetic Algorithm
        ga_solver = GeneticAlgorithmCVRP(
            cvrp_data, population_size=50, generations=100,
            crossover_prob=0.8, mutation_prob=0.1
        )
        ga_stats = ga_solver.run(runs=10)

        results.append({
            "file": file_name,
            "optimal_cost": optimal_cost if optimal_cost is not None else "N/A",
            "greedy_distance": greedy_distance,
            "rand_best": rand_stats["best"],
            "rand_worst": rand_stats["worst"],
            "rand_avg": rand_stats["avg"],
            "rand_std": rand_stats["std"],
            "tabu_best": tabu_stats["best"],
            "tabu_worst": tabu_stats["worst"],
            "tabu_avg": tabu_stats["avg"],
            "tabu_std": tabu_stats["std"],
            "ga_best": ga_stats["best"],
            "ga_worst": ga_stats["worst"],
            "ga_avg": ga_stats["avg"],
            "ga_std": ga_stats["std"]
        })

    # Print results table
    print("\n========== FINAL RESULTS ==========")
    header = ("{:<15}{:<10}{:<10}{:<12}{:<12}{:<12}{:<12}"
              "{:<12}{:<12}{:<12}{:<12}"
              "{:<12}{:<12}{:<12}{:<12}")
    print(header.format(
        "File Name", "Optimal", "Greedy",
        "Rand Best", "Rand Worst", "Rand Avg", "Rand Std",
        "Tabu Best", "Tabu Worst", "Tabu Avg", "Tabu Std",
        "GA Best", "GA Worst", "GA Avg", "GA Std"
    ))
    print("-" * 156)
    for r in results:
        print(header.format(
            r["file"], r["optimal_cost"], f"{r['greedy_distance']:.2f}",
            f"{r['rand_best']:.2f}", f"{r['rand_worst']:.2f}", f"{r['rand_avg']:.2f}", f"{r['rand_std']:.2f}",
            f"{r['tabu_best']:.2f}", f"{r['tabu_worst']:.2f}", f"{r['tabu_avg']:.2f}", f"{r['tabu_std']:.2f}",
            f"{r['ga_best']:.2f}", f"{r['ga_worst']:.2f}", f"{r['ga_avg']:.2f}", f"{r['ga_std']:.2f}"
        ))
    # Append results to CSV
    with open(RESULTS_CSV, mode="a", newline="") as file:
        writer = csv.writer(file)
        for r in results:
            writer.writerow([
                r["file"], r["optimal_cost"], r["greedy_distance"],
                r["rand_best"], r["rand_worst"], r["rand_avg"], r["rand_std"],
                r["tabu_best"], r["tabu_worst"], r["tabu_avg"], r["tabu_std"],
                r["ga_best"], r["ga_worst"], r["ga_avg"], r["ga_std"]
            ])

if __name__ == "__main__":
    main()
