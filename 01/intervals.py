import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

def run_model(data):
    # Convert list of dicts to DataFrame and ensure Quantity is a float.
    df = pd.DataFrame(data)
    df['Quantity'] = df['Quantity'].astype(float)
    if 'Month' in df.columns:
        df = df.sort_values("Month")
    df = df.reset_index(drop=True)
    
    # Create a time series index using a monthly date range.
    ts = pd.Series(df['Quantity'].values,
                   index=pd.date_range(start='2020-01-01', periods=len(df), freq='ME'))
    
    # Define forecast horizon.
    forecast_steps = 12 if len(ts) > 12 else max(1, len(ts) // 5)
    ts_train = ts.iloc[:-forecast_steps]
    ts_test = ts.iloc[-forecast_steps:]
    
    # Fit SARIMAX model.
    model = SARIMAX(ts_train, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12))
    result = model.fit(disp=False)
    
    # Forecast.
    forecast = result.get_forecast(steps=forecast_steps)
    predicted = forecast.predicted_mean
    
    # Custom loss calculation:
    y_true = ts_test.values
    y_pred = predicted.values
    denom = np.where(y_true != 0, y_true, y_true + 1)
    loss = np.mean((y_true - y_pred)**2 / denom)
    
    return loss

def predict_next_year(data):
    # Example prediction method: add fixed increment (e.g. +50 units) to last observation.
    if data.empty or 'Quantity' not in data.columns:
        return 0
    last_val = data['Quantity'].iloc[-1]
    return last_val + 50

if __name__ == "__main__":
    input_history = pd.read_csv('01_input_history.csv')
    prediction = predict_next_year(input_history)
    with open('predictions.csv', 'a') as f:
        f.write("Default,{}\n".format(prediction))