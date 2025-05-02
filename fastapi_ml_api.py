from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Load the trained model
model_path = "model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)
print("✅ Model loaded successfully!")
# Define request model
class InputData(BaseModel):
    features: list[float]  # Expecting a list of numerical features

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (change this in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def home():
    return {"message": "Breast Cancer Prediction API is running!"}

@app.post("/predict")
def predict(data: InputData):
    if len(data.features) != model.n_features_in_:
        return {"error": f"Expected {model.n_features_in_} features, but got {len(data.features)}"}
    
    input_array = np.array(data.features, dtype=np.float64).reshape(1, -1)
    prediction = model.predict(input_array)
    
    return {"prediction": int(prediction[0])}





from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os

# Load the trained model dynamically from the current script directory
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(model_path, "rb") as file:
    model = pickle.load(file)
print("✅ Model loaded successfully!")

# Define request model
class InputData(BaseModel):
    features: list[float]  # Expecting a list of numerical features

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (change this in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
print("Model expects:", model.n_features_in_, "features")

@app.get("/")
def home():
    return {"message": "Breast Cancer Prediction API is running!"}


@app.post("/predict")
def predict(data: InputData):
    if len(data.features) != model.n_features_in_:
        return {"error": f"Expected {model.n_features_in_} features, but got {len(data.features)}"}
    
    input_array = np.array(data.features, dtype=np.float64).reshape(1, -1)
    prediction = model.predict(input_array)
    
    return {"prediction": int(prediction[0])}
