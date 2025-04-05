import pandas as pd
import numpy as np

def preprocess_data(data):
    """
    Preprocess the input data:
    - Remove outliers in 'Quantity' using IQR.
    - Normalize the 'Quantity' column using min-max scaling.
    Returns:
        processed_data: DataFrame with an extra column 'Normalized'.
        scale_params: Tuple (min, max) of filtered data.
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    Q1 = data['Quantity'].quantile(0.25)
    Q3 = data['Quantity'].quantile(0.75)
    IQR = Q3 - Q1
    data_filtered = data[(data['Quantity'] >= Q1 - 1.5 * IQR) &
                         (data['Quantity'] <= Q3 + 1.5 * IQR)]
    if data_filtered.empty:
        return data.copy(), (None, None)
    data_filtered = data_filtered.copy()
    min_val = data_filtered['Quantity'].min()
    max_val = data_filtered['Quantity'].max()
    if max_val == min_val:
        data_filtered.loc[:, 'Normalized'] = data_filtered['Quantity']
        scale_params = (None, None)
    else:
        data_filtered.loc[:, 'Normalized'] = (data_filtered['Quantity'] - min_val) / (max_val - min_val)
        scale_params = (min_val, max_val)
    return data_filtered, scale_params

def predict_next_year(data):
    """
    Predict the next year's value using an autoregressive model (AR(1)) on preprocessed data.
    Steps:
    1. Preprocess data (remove outliers and normalize Quantity).
    2. Fit a linear AR(1) model on the normalized data.
    3. Predict on the normalized scale and revert to original scale.
    If insufficient normalized data exists, returns the last available unfiltered value or 0.
    """
    processed_data, scale_params = preprocess_data(data)
    if processed_data.empty:
        return 0
    if len(processed_data) < 2:
        return processed_data['Quantity'].iloc[-1]
    series = processed_data['Normalized'].values
    x = series[:-1]
    y = series[1:]
    # If nearly constant, skip polyfit.
    if np.allclose(x, x[0]):
        normalized_pred = series[-1]
    else:
        try:
            slope, intercept = np.polyfit(x, y, 1)
            normalized_pred = slope * series[-1] + intercept
        except np.linalg.LinAlgError:
            normalized_pred = series[-1]
    min_val, max_val = scale_params
    if min_val is not None and max_val is not None:
        prediction = normalized_pred * (max_val - min_val) + min_val
    else:
        prediction = normalized_pred
    return prediction

def run_model(data):
    """
    Runs the autoregressive model on the preprocessed data and returns a prediction
    along with outputting a dummy loss computed on the filtered normalized training data.
    """
    # Ensure data is a DataFrame.
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    prediction = predict_next_year(data)
    # For demonstration, compute a dummy loss if sufficient data remains.
    processed_data, _ = preprocess_data(data)
    filtered_data = processed_data[processed_data['Quantity'] > 0]
    if len(filtered_data) < 2:
        loss = 0
    else:
        series = filtered_data['Normalized'].values
        x = series[:-1]
        y = series[1:]
        slope, intercept = np.polyfit(x, y, 1)
        predicted = slope * x + intercept
        loss = np.mean(np.abs(predicted - y))
    print(f"AR model loss for regular category: {loss}")
    return prediction

if __name__ == "__main__":
    # For standalone testing: load historical data from 01_input_history.csv.
    input_history = pd.read_csv('01_input_history.csv')
    prediction = predict_next_year(input_history)
    # Append the prediction to predictions.csv with a default Country, Product, and Month.
    with open('predictions.csv', 'a') as f:
        f.write("Default,Default,Default,{}\n".format(prediction))

