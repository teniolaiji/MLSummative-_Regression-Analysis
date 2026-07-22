import os
import joblib
import numpy as np
import pandas as pd
from enum import Enum
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


# def predict_poverty(continuous_inputs: dict, region: str):
#     """
#     continuous_inputs: {feature_name: value} for the non-region features.
#     region: one region string from REGION_MAP.
#     Returns predicted poverty ratio (% below $3.00/day).
#     """
#     row = dict(continuous_inputs)

#     for col in REGION_COLUMNS:
#         row[col] = 0
#     active = REGION_MAP.get(region)
#     if active is not None:
#         row[active] = 1

#     ordered = [row[name] for name in feature_names]
#     X_new = np.array(ordered, dtype=float).reshape(1, -1)
#     X_scaled = scaler.transform(X_new)

#     return round(float(model.predict(X_scaled)[0]), 2)


# if __name__ == '__main__':
#     example = {
#         'electricity_access': 45.0, 'agri_value_added': 25.0,
#         'health_expenditure': 5.0, 'gdp_growth': 3.0,
#         'gdp_per_capita': 800.0, 'education_expenditure': 4.0,
#         'internet_users': 20.0, 'inflation': 8.0,
#         'life_expectancy': 60.0, 'urban_population': 35.0,
#     }
#     print(predict_poverty(example, 'Sub-Saharan Africa'))
