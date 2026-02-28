import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

print("Training diabetes model...")

data = pd.read_csv("diabetes.csv")

# These columns MUST exist in your CSV
X = data[["Glucose", "BMI", "Age"]]
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pickle.dump(model, open("diabetes_model.pkl", "wb"))

print("Diabetes model trained!")
