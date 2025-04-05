import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('chiavetta/data/01_input_history.csv')

# Focus on a specific subset
data_specific = data[(data['Country'] == "Australia") &
                     (data['Product'] == "CleanSmile Floss Picks")].copy()

# NOTE: We keep rows with Quantity equal to 0 to maintain the seasonal pattern

# Convert Month column from string to datetime
data_specific['Month_dt'] = pd.to_datetime(data_specific['Month'], format='%b%Y')

# Sort the data by date and set it as a time series index
data_specific = data_specific.sort_values('Month_dt')
data_specific.set_index('Month_dt', inplace=True)

# Create a time series with a monthly frequency; 'MS' indicates Month Start
# Fill missing months with 0 (if there are any missing)
ts = data_specific['Quantity'].asfreq('MS').fillna(0)

# Define forecast horizon (e.g., last 12 months as test set)
forecast_steps = 12
ts_train = ts.iloc[:-forecast_steps]
ts_test = ts.iloc[-forecast_steps:]

# Fit a seasonal autoregressive model on the training set
model = SARIMAX(ts_train, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12))
result = model.fit(disp=False)

# Forecast for the test period
forecast = result.get_forecast(steps=forecast_steps)
predicted = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Calculate Mean Squared Error
mse = mean_squared_error(ts_test, predicted)
print(f"Mean Squared Error: {mse}")

# Plot the historical data, the training-test split, and the forecast
plt.figure(figsize=(10,6))
plt.plot(ts_train.index, ts_train, label="Training Observations")
plt.plot(ts_test.index, ts_test, label="Test Observations", color="black")
plt.plot(predicted.index, predicted, label="Forecast", color='blue')
plt.fill_between(forecast_ci.index,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1],
                 color='k', alpha=0.2, label='95% Confidence Interval')
plt.xlabel("Date")
plt.ylabel("Quantity")
plt.title("Seasonal AR Forecast using SARIMAX (Including 0's)")
plt.legend()
plt.show()