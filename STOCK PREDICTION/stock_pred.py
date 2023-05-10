import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
df = pd.read_csv('data.csv', usecols=['Date', 'Close'])

# Normalize the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1,1))

# Define the sequence length
sequence_length = 60

# Create the input and output data
X = []
y = []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])
X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
split_ratio = 0.7
split_index = int(split_ratio * len(X))
X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate the model
mse = np.mean(np.square(predictions - y_test_inv))
rmse = np.sqrt(mse)
print('RMSE: %.2f' % rmse)

# Plot the results
import matplotlib.pyplot as plt
plt.plot(df['Date'][len(df)-len(y_test):], y_test_inv, color='blue', label='Actual Stock Price')
plt.plot(df['Date'][len(df)-len(y_test):], predictions, color='red', label='Predicted Stock Price')
plt.xticks(np.arange(0,len(y_test), step=int(len(y_test)/4)), df['Date'][len(df)-len(y_test):][::int(len(y_test)/4)], rotation=45)
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()
