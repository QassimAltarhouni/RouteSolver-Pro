
import os
import csv
from algorithms.genetic_algorithm import GeneticAlgorithmCVRP
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
    INSTANCES = ["A-n32-k5.vrp", "A-n60-k9.vrp"]
    RUNS_PER_CONFIG = 10

    os.makedirs("results", exist_ok=True)
    result_file = "results/ga_tuning_stepwise.csv"

    with open(result_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Step", "Instance", "Optimal", "Population", "Generations", "Mutation Type",
                         "Crossover Type", "Best", "Worst", "Avg", "Std"])

    for instance in INSTANCES:
        file_path = os.path.join(DATA_FOLDER, instance)
        optimal_path = os.path.join(OPTIMAL_FOLDER, instance)
        cvrp_data = CVRPData(file_path)
        optimal_cost = read_optimal_cost(optimal_path) or "N/A"

        print(f"📦 Instance: {instance}")
        print("🔬 Step 1: Tuning Population Size")

        # Step 1: Tune Population Size
        step1_configs = [
            {"population_size": 30, "generations": 50000, "mutation_type": "swap", "crossover_type": "OX"},
            {"population_size": 50, "generations": 50000, "mutation_type": "swap", "crossover_type": "OX"},
            {"population_size": 100, "generations": 50000, "mutation_type": "swap", "crossover_type": "OX"}
        ]
        step = 1
        best_config = None
        best_avg = float('inf')

        for config in step1_configs:
            solver = GeneticAlgorithmCVRP(cvrp_data,
                                          population_size=config["population_size"],
                                          crossover_prob=0.7,
                                          mutation_prob=0.1,
                                          mutation_type=config["mutation_type"],
                                          crossover_type=config["crossover_type"])
            stats = solver.run(runs=RUNS_PER_CONFIG)
            print(f"POP={config['population_size']} | AVG={stats['avg']:.2f}")
            if stats["avg"] < best_avg:
                best_avg = stats["avg"]
                best_config = config

            with open(result_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([f"Step {step}", instance, optimal_cost, config["population_size"],
                                 config["generations"], config["mutation_type"], config["crossover_type"],
                                 stats["best"], stats["worst"], stats["avg"], stats["std"]])

        print("🔬 Step 2: Tuning Mutation Type")

        # Step 2: Tune Mutation Type
        step2_configs = [
            {**best_config, "mutation_type": "swap"},
            {**best_config, "mutation_type": "inversion"}
        ]
        step += 1
        best_avg = float('inf')

        for config in step2_configs:
            solver = GeneticAlgorithmCVRP(cvrp_data,
                                          population_size=config["population_size"],
                                          crossover_prob=0.8,
                                          mutation_prob=0.1,
                                          mutation_type=config["mutation_type"],
                                          crossover_type=config["crossover_type"])
            stats = solver.run(runs=RUNS_PER_CONFIG)
            print(f"MUT={config['mutation_type']} | AVG={stats['avg']:.2f}")
            if stats["avg"] < best_avg:
                best_avg = stats["avg"]
                best_config = config

            with open(result_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([f"Step {step}", instance, optimal_cost, config["population_size"],
                                 config["generations"], config["mutation_type"], config["crossover_type"],
                                 stats["best"], stats["worst"], stats["avg"], stats["std"]])

        print("🔬 Step 3: Tuning Crossover Type")

        # Step 3: Tune Crossover Type
        step3_configs = [
            {**best_config, "crossover_type": "OX"},
            {**best_config, "crossover_type": "PMX"}
        ]
        step += 1
        best_avg = float('inf')

        for config in step3_configs:
            solver = GeneticAlgorithmCVRP(cvrp_data,
                                          population_size=config["population_size"],
                                          crossover_prob=0.8,
                                          mutation_prob=0.1,
                                          mutation_type=config["mutation_type"],
                                          crossover_type=config["crossover_type"])
            stats = solver.run(runs=RUNS_PER_CONFIG)
            print(f"XO={config['crossover_type']} | AVG={stats['avg']:.2f}")
            if stats["avg"] < best_avg:
                best_avg = stats["avg"]
                best_config = config

            with open(result_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([f"Step {step}", instance, optimal_cost, config["population_size"],
                                 config["generations"], config["mutation_type"], config["crossover_type"],
                                 stats["best"], stats["worst"], stats["avg"], stats["std"]])

        print(f"✅ Best Final Config for {instance}: {best_config}\n")

if __name__ == "__main__":
    main()
