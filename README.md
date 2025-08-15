# 📦 RouteSolver Pro

A modern and efficient **Capacitated Vehicle Routing Problem (CVRP)** solver built for scalable logistics optimization.  

Whether you manage a **small fleet** or coordinate **large-scale delivery operations**, RouteSolver Pro delivers **fast, cost-effective** routing solutions.

---

## 🚀 Key Features
- **Flexible Input** — Works with various fleet and customer configurations.  
- **Built-in Constraints** — Supports distance and capacity restrictions out of the box.  
- **Modular Design** — Easily integrate advanced heuristics or metaheuristics.  
- **Extensible Architecture** — Add new solver strategies without changing the core system.  
- **Visualization Support** *(optional)* — Inspect and analyze generated routes.

---

## ⚙️ Installation

```bash
git clone https://github.com/QassimAltarhouni/RouteSolver-Pro.git
cd route-solver-pro
pip install -r requirements.txt
```

---

## 🧠 Usage

```bash
python solve_cvrp.py --data path/to/data.json --output output/routes.json
```

**Arguments:**
- `--data` — Path to JSON file containing customer locations, demands, and fleet information.  
- `--output` — Path to save the computed routes.

---

## 📂 Project Structure

```
├── solve_cvrp.py        # CLI entry point
├── cvrp/
│   ├── __init__.py
│   ├── solver.py        # Core solver logic
│   ├── heuristics.py    # Heuristic strategies
│   └── utils.py         # Helper functions
├── examples/
│   └── demo.json        # Sample problem
└── README.md
```

---

## 🧪 Testing

```bash
pytest
```

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
