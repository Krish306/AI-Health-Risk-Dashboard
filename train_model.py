import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
data = pd.read_csv("heart.csv")
print("CSV loaded successfully")
print(data.head())  # just to check

# Separate features and target
X = data[[
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]]
y = data["target"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the trained model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved!")
