# datasets given from classification algorithm
# define x and y
# discard part of the dataset with y=0

# loading a dataset
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('chiavetta/data/01_input_history.csv')

data_specific = data[(data['Country'] == "Australia") & (data['Product'] == "BrightBreeze Whitening Booster")]

# drop the 'Country' and 'Product' columns
data_specific = data_specific.drop(['Country', 'Product'], axis=1)

# discarding the part of the dataset with y=0
data_specific = data_specific[data_specific['Quantity'] != 0]

# Convert Month column from string to datetime
data_specific['Month'] = pd.to_datetime(data_specific['Month'], format='%b%Y')

# Convert the datetime to a numeric representation (ordinal)
data_specific['Month'] = data_specific['Month'].map(pd.Timestamp.toordinal)

# define x and y
x = data_specific['Month']
y = data_specific['Quantity']

# splitting the dataset into training and testing sets using simple hold-out method
x_train = x[:int(0.8*len(x))]
x_test = x[int(0.8*len(x)):]
y_train = y[:int(0.8*len(y))]
y_test = y[int(0.8*len(x)):]


# using linear regression as model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train.values.reshape(-1, 1), y_train)
y_pred = model.predict(x_test.values.reshape(-1, 1))

# assessing performance
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
# plot the results
plt.scatter(x_test, y_test, color='black', label='Actual data')
plt.plot(x_test, y_pred, color='blue', linewidth=3, label='Predicted data')
plt.title('Actual vs Predicted')
plt.xlabel('Month (ordinal)')
plt.ylabel('Quantity')
plt.legend()
plt.show()