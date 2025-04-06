import pandas as pd
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus
import os

# Read input files
demand_df = pd.read_csv("../chiavetta/data/02_input_target.csv")
capacity_df = pd.read_csv("../chiavetta/data/02_input_capacity.csv")            # Columns: Country, Monthly Capacity
prod_cost_df = pd.read_csv("../chiavetta/data/03_input_productionCost.csv")        # Columns: Country, Product, Unit Cost
ship_cost_df = pd.read_csv("../chiavetta/data/02_03_input_shipmentsCost_example.csv")  # Columns: Origin, Destination, Unit Cost

# Lists of unique indices (from demand and capacity)
countries_demand = demand_df["Country"].unique()
countries_capacity = capacity_df["Country"].unique()
all_countries = list(set(countries_demand).union(set(countries_capacity)))
products = demand_df["Product"].unique()
months = demand_df["Month"].unique()

# Create dictionaries for demand and capacity
# demand: {(country, product, month): quantity}
demand = { (row["Country"], row["Product"], row["Month"]): row["Quantity"]
           for i, row in demand_df.iterrows() }

# capacity: { (country, month): capacity }
capacity = {}
for i, row in capacity_df.iterrows():
    c = row["Country"]
    cap = row["Monthly Capacity"]
    for m in months:
        capacity[(c, m)] = cap

# Production cost: {(Country, Product): Unit Cost}
production_cost = { (row["Country"], row["Product"]): row["Unit Cost"]
                    for i, row in prod_cost_df.iterrows() }

# Shipping cost: {(Origin, Destination): Unit Cost}
shipping_cost = { (row["Origin"], row["Destination"]): row["Unit Cost"]
                  for i, row in ship_cost_df.iterrows() }

# Create LP problem
prob = LpProblem("Production_Shipment_Optimization", LpMinimize)

# Decision variables: production in country i for product p in month m
prod_vars = { (i, p, m): LpVariable(f"prod_{i}_{p}_{m}", lowBound=0)
              for i in all_countries for p in products for m in months }

# Decision variables: shipments from country i (origin) to country j (destination) for product p in month m
ship_vars = { (i, j, p, m): LpVariable(f"ship_{i}_{j}_{p}_{m}", lowBound=0)
              for i in all_countries for j in all_countries for p in products for m in months }

# Objective: minimize total cost = production cost + shipping cost
prob += (
    lpSum([ production_cost[(i, p)] * prod_vars[(i, p, m)]
            for i in all_countries for p in products for m in months ])
  + lpSum([ shipping_cost.get((i, j), 0) * ship_vars[(i, j, p, m)]
            for i in all_countries for j in all_countries for p in products for m in months ])
), "Total_Costs"

# Constraint 1: For destination country j, product p, month m, incoming shipments equal demand.
for j in countries_demand:
    for p in products:
        for m in months:
            dem = demand.get((j, p, m), 0)
            prob += lpSum([ ship_vars[(i, j, p, m)] for i in all_countries ]) == dem, f"Demand_{j}_{p}_{m}"

# Constraint 2: For origin country i, product p, month m, total shipments out equal production.
for i in all_countries:
    for p in products:
        for m in months:
            prob += lpSum([ ship_vars[(i, j, p, m)] for j in all_countries ]) == prod_vars[(i, p, m)], f"ProductionAllocation_{i}_{p}_{m}"

# Constraint 3: For each country i and month m, total production does not exceed capacity.
for i in all_countries:
    for m in months:
        cap = capacity.get((i, m), 0)
        prob += lpSum([ prod_vars[(i, p, m)] for p in products ]) <= cap, f"Capacity_{i}_{m}"

# Solve the LP
status = prob.solve()
print("Optimization Status:", LpStatus[status])

# Collect solutions into lists
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

# Save result files in 03/output folder
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

production_df.to_csv(os.path.join(output_dir, "03_output_productionPlan.csv"), index=False, float_format="%.15f")
shipments_df.to_csv(os.path.join(output_dir, "03_output_shipments.csv"), index=False, float_format="%.15f")
print("Files written: {}/03_output_productionPlan.csv and {}/03_output_shipments.csv".format(output_dir, output_dir))