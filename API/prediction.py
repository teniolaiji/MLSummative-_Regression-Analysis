import os
import joblib
import numpy as np
import pandas as pd
from enum import Enum
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi import File, UploadFile
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import io


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'best_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
feature_names = joblib.load(os.path.join(BASE_DIR, 'feature_names.pkl'))

REGION_COLUMNS = [f for f in feature_names if f.startswith('reg_')]

# Maps a user-facing region name to the dummy column it activates.
# The baseline region (dropped by drop_first=True) maps to None = all zeros.
REGION_MAP = {
    'East Asia & Pacific': None,   # baseline: all region dummies = 0
    'Europe & Central Asia': 'reg_europe_and_central_asia',
    'Latin America & Caribbean': 'reg_latin_america_and_caribbean',
    'Middle East, North Africa, Afghanistan & Pakistan': 'reg_middle_east_north_africa_afghanistan_and_pakistan',
    'North America': 'reg_north_america',
    'South Asia': 'reg_south_asia',
    'Sub-Saharan Africa': 'reg_sub_saharan_africa',
}


# App + CORS
app = FastAPI(
    title="Poverty Prediction API",
    description="Predicts poverty headcount ratio (% below $3.00/day) from World Development Indicators.",
    version="1.0.0",
)
# CORS: restricted to known origins rather than wildcard "*"

origins = [
    "https://mlsummative-regression-analysis.onrender.com",  # Swagger UI / deployed API
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Region as a vallidated enum


class Region(str, Enum):
    east_asia_pacific = 'East Asia & Pacific'
    europe_central_asia = 'Europe & Central Asia'
    latin_america_caribbean = 'Latin America & Caribbean'
    middle_east_north_africa = 'Middle East, North Africa, Afghanistan & Pakistan'
    north_america = 'North America'
    south_asia = 'South Asia'
    sub_saharan_africa = 'Sub-Saharan Africa'


# Input schema: types + realistic range constraints

class PredictionInput(BaseModel):
    electricity_access: float = Field(..., ge=0, le=100,
                                      description="% of population with electricity")
    agri_value_added: float = Field(..., ge=0,
                                    le=100, description="Agriculture as % of GDP")
    health_expenditure: float = Field(..., ge=0, le=100,
                                      description="Health spending as % of GDP")
    gdp_growth: float = Field(..., ge=-50, le=50,
                              description="Annual GDP growth %")
    gdp_per_capita: float = Field(..., ge=0, le=200000,
                                  description="GDP per capita (current US$)")
    education_expenditure: float = Field(..., ge=0, le=100,
                                         description="Education spending as % of GDP")
    internet_users: float = Field(..., ge=0, le=100,
                                  description="% of population using the internet")
    inflation: float = Field(..., ge=-10, le=100,
                             description="Annual inflation %")
    life_expectancy: float = Field(..., ge=20, le=90,
                                   description="Life expectancy at birth (years)")
    urban_population: float = Field(..., ge=0, le=100,
                                    description="% of population in urban areas")
    region: Region = Field(..., description="World Bank region")

    class Config:
        json_schema_extra = {
            "example": {
                "electricity_access": 45.0, "agri_value_added": 25.0,
                "health_expenditure": 5.0, "gdp_growth": 3.0,
                "gdp_per_capita": 800.0, "education_expenditure": 4.0,
                "internet_users": 20.0, "inflation": 8.0,
                "life_expectancy": 60.0, "urban_population": 35.0,
                "region": "Sub-Saharan Africa"
            }
        }


class PredictionOutput(BaseModel):
    predicted_poverty_ratio: float


# Prediction logic

def build_feature_row(data: PredictionInput):
    row = {
        'electricity_access': data.electricity_access,
        'agri_value_added': data.agri_value_added,
        'health_expenditure': data.health_expenditure,
        'gdp_growth': data.gdp_growth,
        'gdp_per_capita': data.gdp_per_capita,
        'education_expenditure': data.education_expenditure,
        'internet_users': data.internet_users,
        'inflation': data.inflation,
        'life_expectancy': data.life_expectancy,
        'urban_population': data.urban_population,
    }
    # one-hot the region server-side
    for col in REGION_COLUMNS:
        row[col] = 0
    active = REGION_MAP.get(data.region.value)
    if active is not None:
        row[active] = 1

    ordered = [row[name] for name in feature_names]
    return np.array(ordered, dtype=float).reshape(1, -1)


# Endpoints

def root():
    return {"message": "Poverty Prediction API. Visit /docs for Swagger UI."}

@app.get("/")
def root():
    return {"message": "Poverty Prediction API. Visit /docs for Swagger UI."}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    try:
        X_new = build_feature_row(data)
        X_scaled = scaler.transform(X_new)
        prediction = float(model.predict(X_scaled)[0])
        return PredictionOutput(predicted_poverty_ratio=round(prediction, 2))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/retrain")
async def retrain(file: UploadFile = File(...)):
    """
    Upload a CSV of new data to retrain the model.
    The CSV must contain the same feature columns used in training,
    plus the target column 'poverty_ratio'.
    """
    global model, scaler   # replace the in-memory model with the retrained one

    # Only accept CSV files
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400, detail="Please upload a .csv file.")

    try:
        contents = await file.read()
        new_df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Could not read CSV: {str(e)}")

    # Check the required columns are present
    required = feature_names + ['poverty_ratio']
    missing = [c for c in required if c not in new_df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"CSV is missing required columns: {missing}"
        )

    # Prepare data
    X = new_df[feature_names]
    y = new_df['poverty_ratio']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Refit scaler + model on the new data
    new_scaler = StandardScaler().fit(X_train)
    new_model = RandomForestRegressor(random_state=42)
    new_model.fit(new_scaler.transform(X_train), y_train)

    # Evaluate
    preds = new_model.predict(new_scaler.transform(X_test))
    new_rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    # Swap in the new model and persist to disk
    model = new_model
    scaler = new_scaler
    joblib.dump(model, os.path.join(BASE_DIR, 'best_model.pkl'))
    joblib.dump(scaler, os.path.join(BASE_DIR, 'scaler.pkl'))

    return {
        "message": "Model retrained successfully.",
        "rows_used": len(new_df),
        "new_rmse": round(new_rmse, 3)
    }

