import joblib
from sklearn.linear_model import LinearRegression
import pandas as pd

df = pd.DataFrame({
    "area": [1000, 1500, 2000],
    "bedrooms": [2, 3, 4],
    "price": [50, 70, 90]
})

X = df[["area", "bedrooms"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "house_model.pkl")
print("Model saved successfully!")
