from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib

app = FastAPI()

# Load your trained model
model = joblib.load("house_model.pkl")

# Input schema
class Input(BaseModel):
    data: Optional[list] = [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23]

# Prediction endpoint
@app.post("/predict")
def predict(input: Input = Input()):
    pred = model.predict([input.data])
    return {"prediction": float(pred[0])}


# Uvicorn local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
