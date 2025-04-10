import numpy as np  # Import NumPy for efficient matrix calculations

# ==============================
# CVRP Data Handling Class
# ==============================
class CVRPData:
    def __init__(self, file_path):
        """Initialize CVRP data by reading from a file."""
        self.locations = {}  # Dictionary to store node coordinates {node_id: (x, y)}
        self.demands = {}  # Dictionary to store customer demands {node_id: demand}
        self.capacity = 0  # Vehicle capacity (maximum load a truck can carry)
        self.depot = None  # Coordinates of the depot (starting point)
        self.distance_matrix = None  # Matrix to store distances between all locations

        self.load_data(file_path)  # Load data from the input file
        self.compute_distance_matrix()  # Compute the distance matrix

    def load_data(self, file_path):
        """Reads the CVRP instance file and extracts relevant data."""
        with open(file_path, "r") as file:
            lines = file.readlines()

        reading_nodes = False  # Flag to indicate when we are reading location data
        reading_demands = False  # Flag to indicate when we are reading demand data

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue  # Skip empty lines

            if "CAPACITY" in line:
                self.capacity = int(parts[-1])  # Store vehicle capacity

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
                    self.depot = (x, y)  # Store depot location

            elif reading_demands:
                node_id, demand = map(int, parts)
                self.demands[node_id] = demand

    def compute_distance_matrix(self):
        """Computes the Euclidean distance matrix for all locations."""
        num_locations = len(self.locations)
        #print(num_locations)
        self.distance_matrix = np.zeros((num_locations + 1, num_locations + 1)) # all 0s now
        #print(self.distance_matrix)
        for i in self.locations:
            for j in self.locations:
                if i != j:
                    x1, y1 = self.locations[i]
                    x2, y2 = self.locations[j]
                    self.distance_matrix[i, j] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


    def print_data(self):
        """Prints the loaded CVRP data."""
      #  print("\n🚛 Vehicle Capacity:", self.capacity)
      #  print("🏠 Depot Location:", self.depot)
      #  print("📍 Customer Locations (ID, X, Y):", self.locations)
      #  print("📦 Customer Demands:", self.demands)


# Run the script with your file
cvrp = CVRPData("data/A-n60-k9.vrp")  # Ensure the file is placed in "data/"
cvrp.print_data()
