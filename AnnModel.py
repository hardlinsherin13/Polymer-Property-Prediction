import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
data=pd.read_excel('/content/drive/My Drive/PolymerDataset1.xlsx')
data = data[['Eat', 'Eea', 'Egb', 'Ei', 'eps', 'Median - Var', 'Median + Var']]

X = data[['Eat', 'Eea', 'Egb', 'Ei', 'eps']]
y = data[['Median - Var', 'Median + Var']]

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define the model
# Building the ANN Model
model = Sequential()
model.add(Dense(1024, activation='relu', input_dim=X.shape[1]))  # First hidden layer
model.add(Dense(512, activation='relu'))  # Second hidden layer
model.add(Dense(256, activation='relu'))  # Third hidden layer
model.add(Dense(2))  # Output layer with 2 neurons for min and max temperatures

# Compile the model
model.compile(optimizer='adam', loss='mse')
# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=0)
# Make predictions
predictions = model.predict(X_test)

# Calculate the mean squared error
mse_min = mean_squared_error(y_test[0], predictions[0])
mse_max = mean_squared_error(y_test[1], predictions[1])

predictions = scaler_y.inverse_transform(predictions)

print('Mean Squared Error for Minimum Temperature:', mse_min)
print('Mean Squared Error for Maximum Temperature:', mse_max)
print(predictions)
from scipy.stats import f_oneway

# Perform ANOVA
f_statistic, p_value = f_oneway(y_test[0], predictions[0])
print('Mean Squared Error for Minimum Temperature:', mse_min)
print('Mean Squared Error for Maximum Temperature:', mse_max)
print('F-Statistic:', f_statistic)
print('P-Value:', p_value)