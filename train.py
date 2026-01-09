import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json
import os

print("MLOps Lab 2 - Training Pipeline")
print("Student: Kishan (2022BCS0206)")
print("="*50)

# Load dataset
print("\n[1/6] Loading dataset...")
df = pd.read_csv('dataset/winequality-red.csv', sep=';')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Prepare features
X = df.drop('quality', axis=1)
y = df['quality']

print(f"\n[2/6] Features: {X.shape[1]}")
print("\n[3/6] Preprocessing: None")
print("Feature Selection: All features")

# Train-test split
print("\n[4/6] Splitting data (80-20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Train model
print("\n[5/6] Training Decision Tree Regressor (max_depth=5)...")
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)
print("Training complete!")

# Evaluate
print("\n[6/6] Evaluating model...")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics
print("\n" + "="*50)
print("RESULTS")
print(f"MSE: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")
print("="*50)

# Save outputs
print("\nSaving outputs...")
os.makedirs('outputs', exist_ok=True)
with open('outputs/model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved to outputs/model.pkl")

results = {
    "student": "Kishan",
    "roll_number": "2022BCS0206",
    "experiment": "EXP-05",
    "model": "DecisionTreeRegressor",
    "preprocessing": "None",
    "feature_selection": "All features",
    "test_split": 0.2,
    "hyperparameters": {"max_depth": 5},
    "mse": float(mse),
    "r2_score": float(r2)
}

with open('outputs/results.json', 'w') as f:
    json.dump(results, f, indent=4)
print("Results saved to outputs/results.json")

print("\nPipeline completed successfully!")
