print("Script started")

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("heart.csv")

# Features used in app
features = ["age", "trestbps", "chol", "thalach"]
X = data[features]
y = data["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

# Train
pipeline.fit(X_train, y_train)

# Save
with open("heart_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Heart model retrained successfully.")
