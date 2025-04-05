import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# For the neural network (not used now)
# import tensorflow as tf
# from tensorflow.keras.models import Sequential  # type: ignore
# from tensorflow.keras.layers import LSTM, Dense  # type: ignore
# from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator  # type: ignore

# Suppress convergence warnings from SARIMAX
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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
    Predict the next year's value using an advanced SARIMAX model on preprocessed data.
    (Falls back to the last observed Quantity if data is insufficient or model fitting fails.)
    """
    processed_data, scale_params = preprocess_data(data)
    if processed_data.empty or len(processed_data) < 5:
        return processed_data['Quantity'].iloc[-1]
    series = processed_data['Normalized'].values
    try:
        # Fit a SARIMAX(1,0,1) model
        model = SARIMAX(series, order=(1, 0, 1),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        fit_model = model.fit(disp=False)
        forecast_norm = fit_model.forecast(steps=1)[0]
    except Exception:
        forecast_norm = series[-1]
    min_val, max_val = scale_params
    if (min_val is not None) and (max_val is not None):
        prediction = forecast_norm * (max_val - min_val) + min_val
    else:
        prediction = forecast_norm
    return prediction

# Optionally, you may include a grid search to optimize the SARIMAX order:
def find_best_model(series, orders):
    """
    Tries different SARIMAX orders and returns the fitted model with the lowest AIC.
    orders: list of tuples defining the (p,d,q) order.
    """
    best_aic = np.inf
    best_order = None
    best_model = None
    for order in orders:
        try:
            model = SARIMAX(series, order=order,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            fit_model = model.fit(disp=False)
            if fit_model.aic < best_aic:
                best_aic = fit_model.aic
                best_order = order
                best_model = fit_model
        except Exception:
            continue
    return best_model, best_order

def run_model(data):
    """
    Runs an optimized SARIMAX model on the preprocessed data,
    computes in-sample loss via grid search, and returns the one-step-ahead forecast.
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    # Use the SARIMAX forecast
    prediction = predict_next_year(data)
    
    processed_data, _ = preprocess_data(data)
    filtered_data = processed_data[processed_data['Quantity'] > 0]
    if len(filtered_data) < 5:
        loss = 0
    else:
        series = filtered_data['Normalized'].values
        orders = [(1, 0, 1), (1, 1, 1), (2, 0, 2), (2, 1, 2)]
        best_model, best_order = find_best_model(series, orders)
        if best_model is not None:
            forecast_norm = best_model.predict(start=0, end=len(series)-1)
            loss = np.mean(np.abs(forecast_norm - series))
            print(f"Optimized SARIMAX order: {best_order}")
        else:
            loss = np.mean(np.abs(series[1:] - series[:-1]))
    print(f"SARIMAX model loss: {loss}")
    return prediction

if __name__ == "__main__":
    # For standalone testing, load historical data from 01_input_history.csv.
    input_history = pd.read_csv('01_input_history.csv')
    
    # Get forecast using SARIMAX
    sarimax_prediction = predict_next_year(input_history)
    print("SARIMAX forecast:", sarimax_prediction)
    
    # Optionally write the prediction to file
    with open('predictions.csv', 'a') as f:
        f.write("Default,Default,Default,{}\n".format(sarimax_prediction))

