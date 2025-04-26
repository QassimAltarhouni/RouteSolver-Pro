import os
import csv

from algorithms.genetic_algorithm import GeneticAlgorithmCVRP
from algorithms.tabu_algorithm import TabuSearchCVRP
from cvrp_solver import CVRPData


def read_optimal_cost(file_path):
    """
    Read the optimal cost from a solution file.
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
    INSTANCES = ["A-n32-k5.vrp", "A-n60-k9.vrp"]
    RUNS_PER_CONFIG = 10

    os.makedirs("results", exist_ok=True)
    tabu_csv = "results/tabu_tuning_results.csv"
    ga_csv = "results/ga_tuning_results.csv"

    # CSV headers
    with open(tabu_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Instance", "Optimal", "Configuration", "Best", "Worst", "Avg", "Std"])
    with open(ga_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Instance", "Optimal", "Configuration", "Best", "Worst", "Avg", "Std"])

    for instance in INSTANCES:
        file_path = os.path.join(DATA_FOLDER, instance)
        optimal_path = os.path.join(OPTIMAL_FOLDER, instance)
        cvrp_data = CVRPData(file_path)
        optimal_cost = read_optimal_cost(optimal_path) or "N/A"
        print(f"\n📦 Testing Instance: {instance}")

        # Tabu Search tuning
        print("\n🔧 Tuning Tabu Search...\n")
        tabu_configs = [
            {"max_iterations": 200, "tabu_tenure": 5},
            {"max_iterations": 500, "tabu_tenure": 10},
            {"max_iterations": 1000, "tabu_tenure": 15}
        ]
        print(f"{'Optimal':<10}{'Configuration':<40}{'Best':>10}{'Worst':>10}{'Avg':>10}{'Std':>10}")
        for config in tabu_configs:
            solver = TabuSearchCVRP(cvrp_data, **config)
            stats = solver.run(runs=RUNS_PER_CONFIG)
            print(f"{str(optimal_cost):<10}{str(config):<40}"
                  f"{stats['best']:>10.2f}{stats['worst']:>10.2f}"
                  f"{stats['avg']:>10.2f}{stats['std']:>10.2f}")
            with open(tabu_csv, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([instance, optimal_cost, config,
                                 stats['best'], stats['worst'],
                                 stats['avg'], stats['std']])

        # Genetic Algorithm tuning
        print("\n🔧 Tuning Genetic Algorithm...\n")
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
        print(f"{'Optimal':<10}{'Configuration':<40}{'Best':>10}{'Worst':>10}{'Avg':>10}{'Std':>10}")
        for config in ga_configs:
            solver = GeneticAlgorithmCVRP(cvrp_data, **config)
            stats = solver.run(runs=RUNS_PER_CONFIG)
            print(f"{str(optimal_cost):<10}{str(config):<40}"
                  f"{stats['best']:>10.2f}{stats['worst']:>10.2f}"
                  f"{stats['avg']:>10.2f}{stats['std']:>10.2f}")
            with open(ga_csv, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([instance, optimal_cost, config,
                                 stats['best'], stats['worst'],
                                 stats['avg'], stats['std']])

    print("\n✅ Tuning results saved to 'results/' folder!")

if __name__ == "__main__":
    main()
