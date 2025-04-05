import csv
import statistics
import pandas as pd
from classification import leggi_csv_4d, categorize_products
import dismissed  # our custom model using custom loss
import intervals  # our custom model using custom loss
import regular    # our custom model using custom loss

def load_category_map(category_csv):
    """
    Loads the CSV of group classifications into a dictionary.
    Expected columns: Country, Product, Category.
    Returns: dict with keys (Country, Product) and values Category.
    """
    category_map = {}
    df = pd.read_csv(category_csv)
    for _, row in df.iterrows():
        key = (row['Country'], row['Product'])
        category_map[key] = row['Category']
    return category_map

def run_models_on_grouped_data(raw_file, category_map):
    """
    Reads the raw CSV of monthly data, groups it by Country and Product,
    assigns a category from the category_map, and runs the appropriate model.
    Returns a dict mapping category -> list of loss values.
    """
    loss_by_category = {"dismissed": [], "intervals": [], "regular": []}
    df = pd.read_csv(raw_file)
    # Group by country and product.
    groups = df.groupby(['Country', 'Product'])
    for (country, product), group_df in groups:
        key = (country, product)
        if key not in category_map:
            print(f"No category found for {key}. Skipping.")
            continue
        category = category_map[key]
        # Convert group data to list-of-dicts (as expected by the model functions)
        data_group = group_df.to_dict(orient='records')
        loss = None
        if category == "dismissed":
            loss = dismissed.run_model(data_group)
        elif category == "intervals":
            loss = intervals.run_model(data_group)
        elif category == "regular":
            loss = regular.run_model(data_group)
        else:
            print(f"Unknown category {category} for {key}, skipping.")
            continue

        # Print the message including the computed loss.
        if loss is not None:
            print(f"Running model for {country}, {product} (Category: {category}) -> Loss: {loss}")
            loss_by_category[category].append(loss)
        else:
            print(f"Model for {country}, {product} (Category: {category}) returned no result.")
    return loss_by_category

def main():
    raw_file = "../chiavetta/data/01_input_history.csv"
    category_csv = "categories.csv"  # output produced by classification.py

    # Load our mapping of (Country, Product) -> Category.
    category_map = load_category_map(category_csv)
    # Run the appropriate model for each group.
    loss_by_category = run_models_on_grouped_data(raw_file, category_map)
    
    overall_loss = []
    for category, loss_list in loss_by_category.items():
        if loss_list:
            avg_cat = statistics.mean(loss_list)
            overall_loss.extend(loss_list)
            print(f"Average custom loss for {category} model: {avg_cat}")
        else:
            print(f"No loss results for {category} model.")
    
    if overall_loss:
        avg_overall = statistics.mean(overall_loss)
        print(f"Overall Average custom loss: {avg_overall}")
    else:
        print("No custom loss results available.")

if __name__ == "__main__":
    main()