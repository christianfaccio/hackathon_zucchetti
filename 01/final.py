import csv
import statistics
import pandas as pd
# Removed classification import as we use the category_map
# from classification import leggi_csv_4d, categorize_products, classify
import dismissed  # our custom model (always 0)
import intervals   # our custom model (linear regression for target month)
import regular    # our custom model (SARIMAX forecasting)

def load_category_map(category_csv):
    """
    Loads the CSV of group classifications into a dictionary.
    Expected columns: Country, Product, Category.
    Returns: dict with keys (Country, Product) and values Category.
    """
    category_map = {}
    try:
        df = pd.read_csv(category_csv)
        for _, row in df.iterrows():
            key = (row['Country'], row['Product'])
            category_map[key] = row['Category']
    except FileNotFoundError:
        print(f"Error: Category file not found at {category_csv}")
        return None
    except KeyError as e:
        print(f"Error: Missing expected column in {category_csv}: {e}")
        return None
    return category_map

def create_predictions_file(data_file, category_map, output_file, forecast_year=2024):
    """
    Reads the data file, groups by Country and Product, uses the category_map
    to classify each group, then calls the appropriate model's predict function
    to obtain 12-month predictions for the specified forecast_year.
    Predicted hypotheses for each model are written to the output file.
    """
    if category_map is None:
        print("Error: Category map is not loaded. Cannot proceed.")
        return

    try:
        data = pd.read_csv(data_file)
        # Ensure 'Month' column exists
        if 'Month' not in data.columns:
            print(f"Error: 'Month' column not found in {data_file}")
            return
        # Convert Month to datetime objects to reliably find the max date
        # Handle potential errors during conversion
        data['Date'] = pd.to_datetime(data['Month'], format='%b%Y', errors='coerce')
        data.dropna(subset=['Date'], inplace=True) # Drop rows where conversion failed
        
        # Check if data ends in December 2023 as expected by user
        last_date = data['Date'].max()
        if last_date.year != 2023 or last_date.month != 12:
             print(f"Warning: Last date in data is {last_date.strftime('%b%Y')}, not Dec 2023 as expected.")

    except FileNotFoundError:
        print(f"Error: Data file not found at {data_file}")
        return
    except KeyError as e:
         print(f"Error: Missing expected column in {data_file}: {e}")
         return
    except Exception as e:
        print(f"An unexpected error occurred while reading {data_file}: {e}")
        return


    groups = data.groupby(['Country', 'Product'])
    results = []

    # Generate forecast month labels for the target year
    forecast_months = [f"{pd.Timestamp(f'{forecast_year}-{m}-01').strftime('%b')}{forecast_year}" for m in range(1, 13)]

    for (country, product), group_data in groups:
        # Use the pre-loaded category map for classification
        model_choice = category_map.get((country, product), 'unknown') # Default if group not in map

        pred_dict = {}
        try:
            if model_choice == 'dismissed':
                # dismissed.predict_next_year returns 0; create dict for all months.
                 # Pass forecast_year to ensure consistency if needed (though dismissed ignores it)
                pred_val = dismissed.predict_next_year(group_data)
                pred_dict = {m: pred_val for m in forecast_months}
            elif model_choice == 'intervals':
                # Pass forecast_year to the intervals model
                pred_dict = intervals.predict_next_year(group_data, forecast_year)
            elif model_choice == 'regular':
                 # Pass forecast_year to the regular model
                pred_dict = regular.predict_next_year(group_data, forecast_year)
            else:
                # Fallback for unknown categories or groups not in map: predict 0.
                print(f"Warning: Unknown category '{model_choice}' for ({country}, {product}). Predicting 0.")
                pred_dict = {m: 0 for m in forecast_months}

            # Ensure the prediction is a dictionary covering all forecast months
            final_pred_dict = {}
            for m in forecast_months:
                 # Use .get() with default 0 for safety, ensure integer output
                final_pred_dict[m] = int(round(pred_dict.get(m, 0)))

        except Exception as e:
            print(f"Error predicting for ({country}, {product}) using model {model_choice}: {e}")
            # Fallback on error: predict 0 for all months
            final_pred_dict = {m: 0 for m in forecast_months}


        for m in forecast_months:
            results.append({
                'Country': country,
                'Product': product,
                'Month': m,
                'Quantity': final_pred_dict.get(m, 0) # Use get for safety
            })

    predictions_df = pd.DataFrame(results)
    try:
        predictions_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
    except Exception as e:
        print(f"Error writing predictions to {output_file}: {e}")


def main():
    # Use the correct path for the input data
    # Make sure this path is correct relative to where you run the script
    raw_file = "input/01_input_history.csv" # Assuming it's in the 'code' folder relative to execution
    # raw_file = "../chiavetta/data/01_input_history.csv" # Original path from your code

    category_csv = "categories.csv"  # Assuming it's in the 'code' folder

    # Load our mapping of (Country, Product) -> Category.
    category_map = load_category_map(category_csv)

    # Define output file path
    output_file = 'output/01_output_predictions_2179.csv' # Corrected output name

    # Specify the target forecast year
    target_year = 2024

    # Pass the correct data file and the category map
    create_predictions_file(raw_file, category_map, output_file, forecast_year=target_year)

if __name__ == "__main__":
    main()