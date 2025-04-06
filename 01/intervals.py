import pandas as pd
import numpy as np
import calendar

def predict_next_year(data, forecast_year):
    """
    Receives a dataset and returns a dictionary with predicted sales for each month
    of the specified forecast_year. It assumes historical positive sales occur mainly in one month.

    For the target month (month with max historical sales), linear regression
    (Quantity ~ year) predicts the sales for that month in the forecast_year.
    For all other months, the prediction is 0.

    Args:
        data (pd.DataFrame): Historical data for a specific country-product group.
                               Expected columns: 'Month', 'Quantity'.
        forecast_year (int): The year for which to generate predictions (e.g., 2024).

    Returns:
        dict: Keys are forecast month labels (e.g., "Jan2024") and values are predictions.
    """
    # Generate forecast month labels first
    predictions = {f"{calendar.month_abbr[m]}{forecast_year}": 0 for m in range(1, 13)}

    # If dataset is empty, return predictions (all zeros).
    if data.empty:
        return predictions

    # Filter out rows with non-positive sales.
    data_positive = data[data['Quantity'] > 0].copy() # Work on a copy
    if data_positive.empty:
        return predictions # Return zeros if no positive sales history

    # Convert 'Month' column to datetime. Expected format, e.g., "Jan2018".
    # Use errors='coerce' to handle potential invalid date formats gracefully.
    data_positive['date'] = pd.to_datetime(data_positive['Month'], format='%b%Y', errors='coerce')
    data_positive.dropna(subset=['date'], inplace=True) # Remove rows where conversion failed
    if data_positive.empty:
        return predictions # Return zeros if no valid dates found

    # Identify the target month: the month corresponding to the maximum Quantity in positive history.
    try:
        target_row = data_positive.loc[data_positive['Quantity'].idxmax()]
        target_date = target_row['date']
        target_month = target_date.month  # integer between 1 and 12
    except ValueError:
         # This can happen if idxmax() returns NaN (e.g., all quantities were NaN after filtering)
         return predictions # Return zeros if target month cannot be determined


    # Get historical data for the target month.
    df_target = data_positive[data_positive['date'].dt.month == target_month].copy()
    if df_target.empty:
        # Should not happen if target_month was derived from data_positive, but check for safety
        predicted_value = 0
    else:
        # Use the year as the independent variable.
        df_target['year'] = df_target['date'].dt.year
        years = df_target['year'].values
        quantities = df_target['Quantity'].values

        # Ensure no NaN values in quantities used for regression/prediction
        valid_indices = ~np.isnan(quantities)
        years = years[valid_indices]
        quantities = quantities[valid_indices]


        if len(years) < 2:
            # If less than 2 points, predict the last known value for that month. Handle empty case.
             predicted_value = quantities[-1] if len(quantities) > 0 else 0
        else:
            # Fit a linear regression: Quantity = m * year + c
            # Check for sufficient variation in years to avoid singular matrix
            if np.all(years == years[0]):
                 predicted_value = np.mean(quantities) # If all points are from the same year, predict mean
            else:
                try:
                    coeffs = np.polyfit(years, quantities, 1)
                    # Predict for the specific forecast_year passed to the function
                    predicted_value = coeffs[0] * forecast_year + coeffs[1]
                     # Ensure prediction is not negative
                    predicted_value = max(0, predicted_value)
                except np.linalg.LinAlgError:
                    # Handle cases where polyfit fails (e.g., collinearity)
                     predicted_value = np.mean(quantities) # Fallback to mean


    # Update the predictions dictionary for the target month
    target_month_label = f"{calendar.month_abbr[target_month]}{forecast_year}"
    predictions[target_month_label] = predicted_value # Already ensured non-negative

    # Round predictions to nearest integer
    for month_label in predictions:
        predictions[month_label] = int(round(predictions[month_label]))

    return predictions

# Keep the __main__ block for potential individual testing if needed
# but ensure it doesn't interfere when imported by final.py
if __name__ == "__main__":
    # Example usage for testing:
    # Create dummy data consistent with expected input
    history_data = {
        'Month': ['Jan2022', 'Feb2022', 'Jan2023', 'Feb2023', 'Mar2023'],
        'Quantity': [10, 0, 15, 0, 5] # Example where Jan is target month
    }
    input_df = pd.DataFrame(history_data)
    target_forecast_year = 2024
    predictions_output = predict_next_year(input_df, target_forecast_year)

    print(f"Test Predictions for {target_forecast_year}:")
    print(predictions_output)

    # Example writing (optional, adjust paths/logic as needed for testing)
    # output_file_test = 'code/output/intervals_test_output.csv'
    # with open(output_file_test, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['Country', 'Product', 'Month', 'Quantity']) # Header
    #     for month, pred in predictions_output.items():
    #         writer.writerow(['TestCountry', 'TestProduct', month, pred])
    # print(f"Test predictions saved to {output_file_test}")