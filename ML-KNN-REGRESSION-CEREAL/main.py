from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the saved model and scaler
knn_model = joblib.load('app/knn_regression_model.pkl')
scaler = joblib.load('app/scaler.pkl')

app = FastAPI()

# Define the input data format
class CerealFeatures(BaseModel):
    calories: float
    protein: float
    fat: float
    sodium: float
    fiber: float
    carbo: float
    sugars: float
    potass: float
    vitamins: float
    weight: float
    cups: float

@app.post('/predict')
def predict(features: CerealFeatures):
    # Convert input data to a numpy array and reshape it
    data = np.array([
        features.calories, features.protein, features.fat, features.sodium,
        features.fiber, features.carbo, features.sugars, features.potass,
        features.vitamins, features.weight, features.cups
    ]).reshape(1, -1)

    # Scale the data
    scaled_data = scaler.transform(data)

    # Make the prediction
    prediction = knn_model.predict(scaled_data)

    return {'predicted_rating': prediction[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
