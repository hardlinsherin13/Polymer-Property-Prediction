import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
data=pd.read_excel('/content/drive/My Drive/PolymerDataset1.xlsx')
data = data[['Eat', 'Eea', 'Egb', 'Egc', 'Ei', 'eps', 'nc', 'Median - Var', 'Median + Var']]

X = data[['Eat', 'Eea', 'Egb', 'Egc', 'Ei', 'eps', 'nc']]
y_min = data[['Median - Var']]

scaler_X = StandardScaler()
scaler_y_min = StandardScaler()

X = scaler_X.fit_transform(X)
y_min = scaler_y_min.fit_transform(y_min)

X_train, X_test, y_train_min, y_test_min = train_test_split(X, y_min, test_size=0.2, random_state=42)
X_df = pd.DataFrame(X, columns=['Eat', 'Eea', 'Egb', 'Egc', 'Ei', 'eps', 'nc'])

# Create a random forest regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Initialize the RFE selector
selector = RFE(model, n_features_to_select=5, step=1)

# Fit the selector to the data
selector = selector.fit(X_df, y_min)

# Get names of selected features
selected_features = X_df.columns[selector.support_]

# Create a new DataFrame with selected features
X_selected = X_df[selected_features]

# Now X_selected contains only the selected features
print(X_selected)
