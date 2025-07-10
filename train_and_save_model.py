import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import joblib
import os

# Step 1: Load dataset
df = pd.read_csv("data/train.csv")

# Step 2: Use only simple features
features_to_use = [
    "Annual Income",
    "Health Score",
    "Marital Status",
    "Number of Dependents",
    "Credit Score"
]
X = df[features_to_use]
y = df["Premium Amount"]

# Step 3: Define numeric and categorical
num_features = ["Annual Income", "Health Score", "Number of Dependents", "Credit Score"]
cat_features = ["Marital Status"]

# Step 4: Pipelines
num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore", sparse_output=False)
)

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# Step 5: Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline = make_pipeline(preprocessor, LinearRegression())
model_pipeline.fit(X_train, y_train)

# Step 6: Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model_pipeline, "model/insurance_model.pkl")

print("✅ Model saved successfully at model/insurance_model.pkl")

