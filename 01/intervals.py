import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

def preprocess_data(data):
    """
    Preprocess the input data:
    - Remove outliers in 'Quantity' using the IQR method.
    - Normalize the 'Quantity' column using min-max scaling.
    
    Returns:
        processed_data: DataFrame with an extra column 'Normalized'.
        scale_params: Tuple (min_value, max_value) of the filtered data.
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
    """
    Predict the next year's value for the intervals model.
    For instance, add a constant increment (e.g. 0.2 in normalized scale)
    to the last normalized observation and revert to the original scale.
    """
    processed_data, scale_params = preprocess_data(data)
    if processed_data.empty:
        return 0
    if len(processed_data) < 2:
        return processed_data['Quantity'].iloc[-1]
    
    series = processed_data['Normalized'].values
    pred_normalized = series[-1] + 0.2
    
    min_val, max_val = scale_params
    if min_val is not None and max_val is not None:
        prediction = pred_normalized * (max_val - min_val) + min_val
    else:
        prediction = pred_normalized
    return prediction

if __name__ == "__main__":
    input_history = pd.read_csv('input/01_input_history.csv')
    prediction = predict_next_year(input_history)
    with open('output/01_output_predictions_2179.csv', 'a') as f:
        f.write("Default,Default,Default,{}\n".format(prediction))