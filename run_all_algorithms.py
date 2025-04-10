import os
from cvrp_solver import CVRPData
from greedy_algorithm import GreedyCVRP
from random_algorithm import RandomSearchCVRP
from tabu_algorithm import TabuSearchCVRP
from simulated_annealing import SimulatedAnnealingCVRP
from genetic_algorithm import GeneticAlgorithmCVRP

# Paths to data files
DATA_FOLDER = "data"
OPTIMAL_FOLDER = "data/optimal_data"

# List all VRP files in the data directory
vrp_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".vrp")]

# Store results for printing
results = []

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

    # Greedy
    greedy_solver = GreedyCVRP(cvrp_data)
    _, greedy_distance = greedy_solver.run()

    # Random
    random_solver = RandomSearchCVRP(cvrp_data, num_iterations=1000)
    _, best_rd, worst_rd, avg_rd, std_rd = random_solver.run()

    # Tabu
    tabu_solver = TabuSearchCVRP(cvrp_data, max_iterations=1000)
    _, tabu_distance = tabu_solver.run()

    # Simulated Annealing
    sa_solver = SimulatedAnnealingCVRP(
        cvrp_data,
        initial_temp=1000,
        cooling_rate=0.995,
        stopping_temp=2
    )
    _, sa_distance = sa_solver.run()

    # Genetic Algorithm
    ga_solver = GeneticAlgorithmCVRP(
        cvrp_data,
        population_size=50,
        generations=1000,
        crossover_prob=0.8,
        mutation_prob=0.1 # stop condition -- counter number # treshould
    )
    _, ga_distance = ga_solver.run()

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
        "tabu_distance": tabu_distance,
        "sa_distance": sa_distance,
        "ga_distance": ga_distance
    })

# Print results
print("\n========== FINAL RESULTS ==========")
print(f"{'File Name':<15}{'Optimal':<10}{'Greedy':<10}{'Best Random':<15}{'Worst Random':<15}"
      f"{'Avg Random':<15}{'Std Random':<15}{'Tabu':<10}{'SA':<10}{'GA':<10}")
print("-" * 150)

for r in results:
    print(f"{r['file']:<15}{r['optimal_cost']:<10}{r['greedy_distance']:<10.2f}"
          f"{r['best_random_distance']:<15.2f}{r['worst_random_distance']:<15.2f}"
          f"{r['avg_random_distance']:<15.2f}{r['std_random_distance']:<15.2f}"
          f"{r['tabu_distance']:<10.2f}{r['sa_distance']:<10.2f}{r['ga_distance']:<10.2f}")
