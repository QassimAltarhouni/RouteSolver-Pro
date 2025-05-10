import os
import csv
import time

from algorithms.genetic_algorithm import GeneticAlgorithmCVRP
from algorithms.greedy_algorithm import GreedyCVRP
from algorithms.random_algorithm import RandomSearchCVRP
from algorithms.tabu_algorithm import TabuSearchCVRP
from cvrp_solver import CVRPData


def read_optimal_cost(file_path):
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

    # ✅ Overwrite best_routes.txt ONCE at the start
    with open("results/best_routes.txt", "w", encoding="utf-8") as f:
        f.write("🆕 Best Routes per File\n")
        f.write("=" * 50 + "\n")

    with open(RESULTS_CSV, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "File Name", "Optimal", "Greedy",
            "Rand Best", "Rand Worst", "Rand Avg", "Rand Std",
            "Tabu Best", "Tabu Worst", "Tabu Avg", "Tabu Std",
            "GA Best", "GA Worst", "GA Avg", "GA Std"
        ])

    results = []

    for idx, file_name in enumerate(vrp_files, 1):
        print(f"\n🔄 Processing file {idx}/{len(vrp_files)}: {file_name}")
        file_path = os.path.join(DATA_FOLDER, file_name)
        optimal_file = os.path.join(OPTIMAL_FOLDER, file_name)
        cvrp_data = CVRPData(file_path)
        optimal_cost = read_optimal_cost(optimal_file)

        # Run Greedy
        print("➡️ Running Greedy...")
        start = time.time()
        greedy_solver = GreedyCVRP(cvrp_data)
        greedy_routes, greedy_distance = greedy_solver.run()
        print(f"✅ Greedy Done in {time.time() - start:.2f}s")

        # Run Random Search
        print("➡️ Running Random Search...")
        start = time.time()
        random_solver = RandomSearchCVRP(cvrp_data, max_fitness_evals=5000)
        rand_stats = random_solver.run_multiple(runs=10)
        print(f"✅ Random Search Done in {time.time() - start:.2f}s")

        # Run Tabu Search
        print("➡️ Running Tabu Search...")
        start = time.time()
        tabu_solver = TabuSearchCVRP(cvrp_data, max_iterations=100, neighbor_sample_size=50)
        tabu_stats = tabu_solver.run(runs=10)
        print(f"✅ Tabu Search Done in {time.time() - start:.2f}s")

        # Run Genetic Algorithm
        print("➡️ Running Genetic Algorithm...")
        start = time.time()
        ga_solver = GeneticAlgorithmCVRP(
            cvrp_data, population_size=50, generations=100,
            crossover_prob=0.9, mutation_prob=0.1
        )
        ga_stats = ga_solver.run(runs=10)
        print(f"✅ GA Done in {time.time() - start:.2f}s")

        # ✅ Append current file's results to best_routes.txt
        with open("results/best_routes.txt", "a", encoding="utf-8") as f:
            f.write(f"\n📁 File: {file_name}\n")
            f.write(f"Greedy Best Cost: {greedy_distance:.2f}\nRoutes: {greedy_routes}\n")
            f.write(f"Random Best Cost: {rand_stats['best']:.2f}\nRoutes: {rand_stats['split_routes']}\n")
            f.write(f"Tabu Best Cost: {tabu_stats['best']:.2f}\nRoutes: {tabu_stats['split_routes']}\n")
            f.write(f"GA Best Cost: {ga_stats['best']:.2f}\nRoutes: {ga_stats['split_routes']}\n")
            f.write("---------------------------------------------------\n")

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

    print("\n📝 Writing Results Table...")
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

    # Save results to CSV
    with open(RESULTS_CSV, mode="a", newline="") as file:
        writer = csv.writer(file)
        for r in results:
            writer.writerow([
                r["file"], r["optimal_cost"], r["greedy_distance"],
                r["rand_best"], r["rand_worst"], r["rand_avg"], r["rand_std"],
                r["tabu_best"], r["tabu_worst"], r["tabu_avg"], r["tabu_std"],
                r["ga_best"], r["ga_worst"], r["ga_avg"], r["ga_std"]
            ])

    print("\n✅ All files processed and results saved!")


if __name__ == "__main__":
    main()
