from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load your trained model
model = joblib.load("house_model.pkl")

# API endpoint (JSON input)
from pydantic import BaseModel
from typing import Optional

class Input(BaseModel):
    data: Optional[list] = [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23]

@app.post("/predict")
def predict(input: Input = Input()):
    pred = model.predict([input.data])
    return {"prediction": float(pred[0])}

# UI endpoint (form input)
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "price": None})

@app.post("/predict-ui", response_class=HTMLResponse)
def predict_ui(request: Request, area: float = Form(...), bedrooms: int = Form(...)):
    prediction = model.predict([[area, bedrooms]])[0]
    return templates.TemplateResponse("index.html", {"request": request, "price": round(prediction, 2)})

# Uvicorn local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
