import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("stroke.csv")

features = ["age", "hypertension", "heart_disease", "avg_glucose_level"]

X = data[features].fillna(0)
y = data["stroke"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=1000, class_weight="balanced")

model.fit(X_train, y_train)

with open("stroke_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Stroke model retrained with class balancing.")
