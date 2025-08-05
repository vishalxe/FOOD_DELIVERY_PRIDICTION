import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load dataset
df = pd.read_csv("dataset/food_delivery.csv")

# Drop missing values
df.dropna(inplace=True)

# Define features and target
X = df.drop(columns=['Order_ID', 'Delivery_Time_min'])
y = df['Delivery_Time_min']

# Categorical and numerical columns
categorical = ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
numerical = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs']

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
], remainder='passthrough')

# Final pipeline
pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "model/rf_model.pkl")
print("✅ Model trained and saved successfully.")
