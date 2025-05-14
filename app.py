from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the specific model
MODEL_PATH = "models/v2.pkl"  # You can change this as needed
ENCODER_PATH="models/lib/encoder.pkl"
SCALER_PATH="models/lib/scaler.pkl"
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)

app = FastAPI()

# Allow CORS for localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Input schema matching your features
class CarFeatures(BaseModel):
    model_year: int
    mileage: float
    number_of_doors: int
    first_owner: bool
    tax_horsepower: float
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
    car_age: int
    transmission: str
    fuel_type: str
    brand: str
    model: str
    origin: str
    condition: str

@app.post("/predict")
def predict(car: CarFeatures):
    # Convert input to DataFrame for better handling
    transformedData = transformData(car)
    
    # Make prediction
    prediction = model.predict(transformedData)
    
    return {"prediction": float(prediction[0])}



def transformData(data):

    data = pd.DataFrame([data.dict()]) 

    categorical_cols = [
        'transmission',
        'fuel_type',
        'brand',
        'model',
        'origin',
        'condition',
    ]

    data_encoded = encoder.transform(data[categorical_cols])

    # Create DataFrame with proper column names
    data_encoded = pd.DataFrame(data_encoded, 
                            columns=encoder.get_feature_names_out(categorical_cols),
                            index=data.index)  # keep index aligned with original df

    # Optionally, concatenate back with the original dataframe (without the original categorical columns)
    data = pd.concat([data.drop(columns=categorical_cols), data_encoded], axis=1)

    data[['mileage', 'tax_horsepower', 'car_age', 'number_of_doors']] = scaler.transform(
        data[['mileage', 'tax_horsepower', 'car_age', 'number_of_doors']]
    )
    return data
