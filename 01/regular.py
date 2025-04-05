import pandas as pd
from sklearn.linear_model import LinearRegression
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
        print("Not enough data for regular model.")
        return None

    # Split 80/20.
    train_size = int(0.8 * len(df))
    x_train = x.iloc[:train_size]
    x_test = x.iloc[train_size:]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]
    
    # Linear Regression model.
    model = LinearRegression()
    model.fit(x_train.values.reshape(-1, 1), y_train)
    y_pred = model.predict(x_test.values.reshape(-1, 1))
    
    # Custom loss calculation:
    y_true = y_test.values
    denom = np.where(y_true != 0, y_true, y_true + 1)
    loss = np.mean((y_true - y_pred)**2 / denom)
    
    return loss

