import numpy as np

class CVRPData:
    """
    Represents a Capacitated Vehicle Routing Problem (CVRP) instance.
    Reads node coordinates, demands, and vehicle capacity from a file,
    and computes the distance matrix.
    """
    def __init__(self, file_path):
        """
        Initialize CVRP data by reading from a file and computing distances.
        :param file_path: Path to the CVRP instance file.
        """
        self.locations = {}       # {node_id: (x, y)} for all nodes (including depot)
        self.demands = {}         # {node_id: demand} for each customer
        self.capacity = 0         # Vehicle capacity
        self.depot = None         # Coordinates of the depot (node 1)
        self.distance_matrix = None

        self.load_data(file_path)
        self.compute_distance_matrix()

    def load_data(self, file_path):
        """
        Reads the CVRP instance from a file (TSPLIB format).
        Expects sections: CAPACITY, NODE_COORD_SECTION, DEMAND_SECTION.
        """
        with open(file_path, 'r') as file:
            lines = file.readlines()

        reading_nodes = False
        reading_demands = False

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            if "CAPACITY" in line:
                self.capacity = int(parts[-1])
            elif "NODE_COORD_SECTION" in line:
                reading_nodes = True
                reading_demands = False
            elif "DEMAND_SECTION" in line:
                reading_nodes = False
                reading_demands = True
            elif "DEPOT_SECTION" in line:
                reading_nodes = False
                reading_demands = False
            elif reading_nodes:
                node_id, x, y = map(int, parts)
                self.locations[node_id] = (x, y)
                if node_id == 1:
                    self.depot = (x, y)
            elif reading_demands:
                node_id, demand = map(int, parts)
                self.demands[node_id] = demand

    def compute_distance_matrix(self):
        """
        Computes the Euclidean distance matrix for all nodes.
        """
        num_locations = len(self.locations)
        # +1 because nodes are 1-based (we skip index 0)
        self.distance_matrix = np.zeros((num_locations + 1, num_locations + 1))
        for i in self.locations:
            x1, y1 = self.locations[i]
            for j in self.locations:
                if i == j:
                    continue
                x2, y2 = self.locations[j]
                # Euclidean distance
                self.distance_matrix[i, j] = np.hypot(x1 - x2, y1 - y2)

    def print_data(self):
        """Prints the loaded CVRP data."""
      #  print("\n🚛 Vehicle Capacity:", self.capacity)
      #  print("🏠 Depot Location:", self.depot)
      #  print("📍 Customer Locations (ID, X, Y):", self.locations)
      #  print("📦 Customer Demands:", self.demands)


# Run the script with your file
cvrp = CVRPData("data/A-n60-k9.vrp")  # Ensure the file is placed in "data/"
cvrp.print_data()
