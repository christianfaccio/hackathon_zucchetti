import pandas as pd
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus
import os

# Read input files
demand_df = pd.read_csv("../chiavetta/data/02_input_target.csv")
capacity_df = pd.read_csv("../chiavetta/data/02_input_capacity.csv")  # Columns: Country, Monthly Capacity

# Lists of unique indices
countries_demand = demand_df["Country"].unique()
countries_capacity = capacity_df["Country"].unique()
all_countries = list(set(countries_demand).union(set(countries_capacity)))
products = demand_df["Product"].unique()
months = demand_df["Month"].unique()

# Create dictionaries for demand and capacity
# demand: {(country, product, month): quantity}
demand = {(row["Country"], row["Product"], row["Month"]): row["Quantity"]
          for i, row in demand_df.iterrows()}

# capacity: { (country, month): capacity } assuming capacity applies monthly and country‐wide
capacity = {}
for i, row in capacity_df.iterrows():
    # Assuming capacity_df has columns: "Country" and "Monthly Capacity"
    c = row["Country"]
    cap = row["Monthly Capacity"]
    for m in months:
        capacity[(c, m)] = cap

# Create LP problem
prob = LpProblem("Production_Shipment_Optimization", LpMinimize)

# Decision variables: production produced in country i for product p in month m
prod_vars = {(i, p, m): LpVariable(f"prod_{i}_{p}_{m}", lowBound=0)
             for i in all_countries for p in products for m in months}

# Decision variables: shipments from country i (origin) to country j (destination)
ship_vars = {(i, j, p, m): LpVariable(f"ship_{i}_{j}_{p}_{m}", lowBound=0)
             for i in all_countries for j in all_countries 
             for p in products for m in months}

# Objective: minimize total inter-country shipments (i.e. where origin != destination)
# Ships produced locally incur no cost.
prob += lpSum([ship_vars[(i,j,p,m)] 
               for i in all_countries for j in all_countries 
               for p in products for m in months if i != j]), "Minimize_Inter_Country_Shipments"

# Constraint 1: For each destination country, product, month, shipments in must equal demand.
for j in countries_demand:
    for p in products:
        for m in months:
            # Some (j,p,m) might not be in demand if demand is zero; use get with default 0.
            dem = demand.get((j, p, m), 0)
            prob += lpSum([ship_vars[(i, j, p, m)] for i in all_countries]) == dem, f"Demand_{j}_{p}_{m}"

# Constraint 2: For each origin country, product, month, total shipments out equals production in that country.
for i in all_countries:
    for p in products:
        for m in months:
            prob += lpSum([ship_vars[(i, j, p, m)] for j in all_countries]) == prod_vars[(i, p, m)], f"ProductionAllocation_{i}_{p}_{m}"

# Constraint 3: For each country and month, total production cannot exceed its capacity.
for i in all_countries:
    for m in months:
        # If a country does not have an entry, assume capacity is 0.
        cap = capacity.get((i, m), 0)
        prob += lpSum([prod_vars[(i, p, m)] for p in products]) <= cap, f"Capacity_{i}_{m}"

# Solve the LP
status = prob.solve()
print("Optimization Status:", LpStatus[status])

# Collect solution into dataframes
production_list = []
shipments_list = []
for (i, p, m), var in prod_vars.items():
    val = var.varValue if var.varValue is not None else 0
    if val > 0:
        production_list.append({"Country": i, "Product": p, "Month": m, "Quantity": val})

for (i, j, p, m), var in ship_vars.items():
    val = var.varValue if var.varValue is not None else 0
    if val > 0:
        shipments_list.append({"Origin": i, "Destination": j, "Product": p, "Month": m, "Quantity": val})

production_df = pd.DataFrame(production_list)
shipments_df = pd.DataFrame(shipments_list)

# Save result files
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

production_df.to_csv(os.path.join(output_dir, "optimal_production.csv"), index=False, float_format="%.15f")
shipments_df.to_csv(os.path.join(output_dir, "optimal_shipments.csv"), index=False, float_format="%.15f")
print("Files written: {}/optimal_production.csv and {}/optimal_shipments.csv".format(output_dir, output_dir))