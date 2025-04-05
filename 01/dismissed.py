import pandas as pd
from sklearn.dummy import DummyRegressor
import numpy as np

def run_model(data):
    # Convert list of dicts to DataFrame and ensure Quantity is a float.
    df = pd.DataFrame(data)
    df['Quantity'] = df['Quantity'].astype(float)
    if 'Month' in df.columns:
        df = df.sort_values("Month")
    df = df.reset_index(drop=True)
    
    # Use index as the feature.
    x = pd.Series(range(len(df)))
    y = df['Quantity']
    
    if len(df) < 5:
        print("Not enough data for dismissed model.")
        return None

    # Split 80/20.
    train_size = int(0.8 * len(df))
    x_train = x.iloc[:train_size]
    x_test = x.iloc[train_size:]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]
    
    # Dummy model that always predicts 0.
    model = DummyRegressor(strategy="constant", constant=0)
    model.fit(x_train.values.reshape(-1, 1), y_train)
    y_pred = model.predict(x_test.values.reshape(-1, 1))
    
    # Custom loss calculation:
    y_true = y_test.values
    # If y_true is 0 then denominator becomes y_true + 1 (i.e. 1); otherwise y_true.
    denom = np.where(y_true != 0, y_true, y_true + 1)
    loss = np.mean((y_true - y_pred)**2 / denom)
    
    return loss

def predict_next_year(data):
    # Compute prediction without splitting dataset or loss error.
    # Example: if "Quantity" exists, use its last value increased by 10%.
    if data.empty or 'Quantity' not in data.columns:
        return 0
    last_val = data['Quantity'].iloc[-1]
    return last_val * 1.1

if __name__ == "__main__":
    # For standalone testing: load history from 01_input_history.csv
    input_history = pd.read_csv('01_input_history.csv')
    prediction = predict_next_year(input_history)
    # Append the prediction to predictions.csv with default Country
    with open('predictions.csv', 'a') as f:
        f.write("Default,{}\n".format(prediction))

