from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the specific model
MODEL_PATH = "models/v1.pkl"  # You can change this as needed
model = joblib.load(MODEL_PATH)

app = FastAPI()

# Input schema matching your features
class CarFeatures(BaseModel):
    model_year: int
    transmission: str
    fuel_type: str
    mileage: float
    brand: str
    model: str
    number_of_doors: int
    origin: str
    first_owner: bool
    tax_horsepower: float
    condition: str
    abs: bool
    airbags: int
    multimedia: bool
    backup_camera: bool
    air_conditioning: bool
    esp: bool
    aluminum_wheels: bool
    speed_limiter: bool
    onboard_computer: bool
    parking_sensors: bool
    cruise_control: bool
    leather_seats: bool
    navigation_gps: bool
    sunroof: bool
    remote_central_locking: bool
    power_windows: bool

@app.post("/predict")
def predict(car: CarFeatures):
    # Convert input to DataFrame for better handling
    transformedData = transformData(car)
    
    # Make prediction
    prediction = model.predict(transformedData)
    
    return {"prediction": float(prediction[0])}



def transformData(data):
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])

    return pd.get_dummies(input_df, columns=[
    'transmission',
    'fuel_type',
    'brand',
    'model',
    'origin',
    'condition',
    'first_owner'
], drop_first=True)