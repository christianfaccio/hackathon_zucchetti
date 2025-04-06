import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import calendar
from statsmodels.tools.sm_exceptions import ConvergenceWarning, EstimationWarning

# Suppress specific warnings from SARIMAX to avoid cluttering output
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=EstimationWarning)
warnings.filterwarnings("ignore", category=UserWarning) # Catches freq='None' warnings

def preprocess_data_iqr(data):
    """
    Preprocess the input data: Remove outliers in 'Quantity' using IQR.
    Args:
        data (pd.DataFrame): Input data with 'Quantity' column.
    Returns:
        pd.DataFrame: Data with outliers removed based on IQR.
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        return pd.DataFrame(columns=data.columns if hasattr(data, 'columns') else []) # Return empty frame if input is bad

    data_copy = data.copy()
    Q1 = data_copy['Quantity'].quantile(0.25)
    Q3 = data_copy['Quantity'].quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds, handle cases where IQR is 0
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter data - ensuring bounds are sensible
    if IQR == 0:
      # If IQR is 0, it means 50% or more data points are the same value.
      # Keep only data points equal to the median in this case.
      median_val = data_copy['Quantity'].median()
      # Ensure median_val is not NaN before filtering
      if pd.isna(median_val):
          data_filtered = data_copy # Or return empty frame if median is NaN?
      else:
          data_filtered = data_copy[data_copy['Quantity'] == median_val]
    else:
      data_filtered = data_copy[(data_copy['Quantity'] >= lower_bound) &
                                (data_copy['Quantity'] <= upper_bound)]

    return data_filtered


def predict_next_year(data, forecast_year):
    """
    Predict the quantity for each month of the specified forecast_year using SARIMAX.
    Filters for positive Quantity, removes outliers using IQR, fits SARIMAX.

    Args:
        data (pd.DataFrame): Historical data for a specific country-product group.
                               Expected columns: 'Month', 'Quantity'.
        forecast_year (int): The year for which to generate predictions (e.g., 2024).

    Returns:
        dict: Keys are forecast month labels (e.g., "Jan2024") and values are predictions.
              Returns zeros if insufficient data or model fails.
    """
    predictions = {f"{calendar.month_abbr[m]}{forecast_year}": 0 for m in range(1, 13)}
    min_obs_required = 24 # Minimum observations for SARIMAX with seasonal order 12

    if data.empty:
        return predictions

    # Filter data: Only positive Quantity values and valid dates
    data_positive = data[data['Quantity'] > 0].copy()
    if data_positive.empty:
         return predictions

    data_positive['date'] = pd.to_datetime(data_positive['Month'], format='%b%Y', errors='coerce')
    data_positive.dropna(subset=['date', 'Quantity'], inplace=True) # Drop rows with NA date or Quantity
    if data_positive.empty:
        return predictions

    # Sort data by date to ensure correct time series order
    data_positive.sort_values('date', inplace=True)

    # Optional: Remove outliers using IQR (consider if needed for your data)
    data_processed = preprocess_data_iqr(data_positive)
    if data_processed.empty:
         return predictions

    # Set the date as index and ensure monthly frequency ('MS')
    try:
        # Create a proper DatetimeIndex
        data_processed = data_processed.set_index('date')

        # Ensure the index is monotonically increasing before resampling
        if not data_processed.index.is_monotonic_increasing:
             data_processed = data_processed.sort_index()

        # Resample to monthly frequency ('MS'). This handles potential missing months.
        data_processed = data_processed.asfreq('MS')

        # Check for NaNs introduced by asfreq if the original data had gaps
        if data_processed['Quantity'].isnull().any():
             print(f"Warning: Monthly resampling introduced NaNs for group ({data.iloc[0]['Country'] if 'Country' in data.columns else 'N/A'}, {data.iloc[0]['Product'] if 'Product' in data.columns else 'N/A'}) due to gaps in data. Attempting to interpolate.")
             # Option: Interpolate NaNs using time method (suitable for time series)
             data_processed['Quantity'] = data_processed['Quantity'].interpolate(method='time')
             # Drop any remaining NaNs (e.g., at the start/end after interpolation)
             data_processed.dropna(subset=['Quantity'], inplace=True)

        y = data_processed['Quantity'].astype(float) # Ensure numeric type

        # Check again if enough data remains after processing/resampling
        if len(y) < min_obs_required:
            print(f"Warning: Insufficient data points ({len(y)} < {min_obs_required}) after resampling for SARIMAX model for group ({data.iloc[0]['Country'] if 'Country' in data.columns else 'N/A'}, {data.iloc[0]['Product'] if 'Product' in data.columns else 'N/A'}). Returning zeros.")
            return predictions

    except Exception as e:
        print(f"Warning: Could not set DatetimeIndex with frequency 'MS' for group ({data.iloc[0]['Country'] if 'Country' in data.columns else 'N/A'}, {data.iloc[0]['Product'] if 'Product' in data.columns else 'N/A'}). Error: {e}. Returning zeros.")
        return predictions

    # Check if the index now has a frequency (it should after asfreq)
    if y.index.freq is None:
        # This shouldn't ideally happen after asfreq, but check just in case
        print(f"Warning: Index frequency could not be set to 'MS' for group ({data.iloc[0]['Country'] if 'Country' in data.columns else 'N/A'}, {data.iloc[0]['Product'] if 'Product' in data.columns else 'N/A'}). Statsmodels might still raise warnings.")


    forecast_values = np.zeros(12) # Initialize forecast array

    try:
        # Fit SARIMAX model - Pass the Series y which now has a DatetimeIndex with frequency
        model = SARIMAX(y,
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, 12),
                        simple_differencing=False) # Often recommended with SARIMAX

        results = model.fit(disp=False) # disp=False suppresses convergence output

        # Forecast for the next year (12 months)
        # The forecast index will be based on the frequency of the input 'y' index
        forecast = results.get_forecast(steps=12)
        forecast_values = forecast.predicted_mean.values # Get numpy array

        # Ensure predictions are not negative
        forecast_values = np.maximum(0, forecast_values)

    except np.linalg.LinAlgError:
        print(f"Warning: SARIMAX failed due to Linear Algebra Error (often singular matrix) for group ({data.iloc[0]['Country'] if 'Country' in data.columns else 'N/A'}, {data.iloc[0]['Product'] if 'Product' in data.columns else 'N/A'}). Returning zeros.")
        # Keep forecast_values as zeros
    except ValueError as ve:
         print(f"Warning: SARIMAX failed due to ValueError (check data, orders) for group ({data.iloc[0]['Country'] if 'Country' in data.columns else 'N/A'}, {data.iloc[0]['Product'] if 'Product' in data.columns else 'N/A'}). Error: {ve}. Returning zeros.")
         # Keep forecast_values as zeros
    except Exception as e:
        # Catch any other unexpected errors during model fitting/forecasting
        print(f"Warning: An unexpected error occurred during SARIMAX for group ({data.iloc[0]['Country'] if 'Country' in data.columns else 'N/A'}, {data.iloc[0]['Product'] if 'Product' in data.columns else 'N/A'}): {e}. Returning zeros.")
        # Keep forecast_values as zeros


    # Build the dictionary of predictions keyed by month labels
    # Use the forecast_values array directly
    for i, m in enumerate(range(1, 13)):
         month_label = f"{calendar.month_abbr[m]}{forecast_year}"
         # Round predictions to nearest integer
         # Check if forecast_values has the expected length
         if i < len(forecast_values):
             predictions[month_label] = int(round(forecast_values[i]))
         else:
             # Handle unexpected case where forecast is shorter than 12 steps
             predictions[month_label] = 0


    return predictions


# Keep the __main__ block for potential individual testing
if __name__ == "__main__":
    # Example usage for testing:
     # Create dummy data spanning multiple years for SARIMAX testing
    dates = pd.date_range(start='2020-01-01', periods=36, freq='MS')
    # Simple seasonal pattern + trend + noise
    quantities = [10 + i//12 * 2 + (i%12)*1.5 + np.random.rand()*5 for i in range(36)]
    quantities = [max(0, q) for q in quantities] # Ensure non-negative

    history_data = {
        'Month': dates.strftime('%b%Y'),
        'Quantity': quantities,
        # Add dummy Country/Product for testing warnings
        'Country': ['TestCountry'] * 36,
        'Product': ['TestProduct'] * 36
    }
    input_df = pd.DataFrame(history_data)

    # Test case with gap
    input_df_gap = input_df.drop(input_df.index[15])


    target_forecast_year = 2023 # Predict year after data ends (data ends Dec 2022)

    print("\n--- Testing with full data ---")
    predictions_output = predict_next_year(input_df, target_forecast_year)
    print(f"Test Predictions for {target_forecast_year}:")
    print(predictions_output)

    print("\n--- Testing with data gap ---")
    predictions_output_gap = predict_next_year(input_df_gap, target_forecast_year)
    print(f"Test Predictions for {target_forecast_year} (with gap):")
    print(predictions_output_gap)