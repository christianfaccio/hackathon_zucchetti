import pandas as pd
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

def predict_next_year(data):
    """
    Predict the next year's value for the dismissed model.
    For example, take the last normalized value, apply a 15% increase,
    and convert back to the original scale.
    """
    processed_data, scale_params = preprocess_data(data)
    if processed_data.empty:
        return 0
    if len(processed_data) < 2:
        return processed_data['Quantity'].iloc[-1]
    
    # For dismissed model, increase the last normalized observation by 15%
    series = processed_data['Normalized'].values
    pred_normalized = series[-1] * 1.15
    
    min_val, max_val = scale_params
    if min_val is not None and max_val is not None:
        prediction = pred_normalized * (max_val - min_val) + min_val
    else:
        prediction = pred_normalized
    return prediction

def run_model(data):
    """
    Runs the dismissed model on preprocessed data and outputs a dummy loss.
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    prediction = predict_next_year(data)
    processed_data, _ = preprocess_data(data)
    if len(processed_data) < 2:
        loss = 0
    else:
        series = processed_data['Normalized'].values
        loss = np.abs(series[-1] - series[-2])
    print(f"Dismissed model loss for dismissed category: {loss}")
    return prediction

if __name__ == "__main__":
    input_history = pd.read_csv('01_input_history.csv')
    prediction = predict_next_year(input_history)
    with open('predictions.csv', 'a') as f:
        f.write("Default,Default,Default,{}\n".format(prediction))

