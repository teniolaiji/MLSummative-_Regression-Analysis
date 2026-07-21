import joblib
import numpy as np
import os

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

def predict_poverty(continuous_inputs: dict, region: str):
    """
    continuous_inputs: {feature_name: value} for the non-region features.
    region: one region string from REGION_MAP.
    Returns predicted poverty ratio (% below $3.00/day).
    """
    row = dict(continuous_inputs)

    for col in REGION_COLUMNS:
        row[col] = 0
    active = REGION_MAP.get(region)
    if active is not None:
        row[active] = 1

    ordered = [row[name] for name in feature_names]
    X_new = np.array(ordered, dtype=float).reshape(1, -1)
    X_scaled = scaler.transform(X_new)

    return round(float(model.predict(X_scaled)[0]), 2)


if __name__ == '__main__':
    example = {
        'electricity_access': 45.0, 'agri_value_added': 25.0,
        'health_expenditure': 5.0, 'gdp_growth': 3.0,
        'gdp_per_capita': 800.0, 'education_expenditure': 4.0,
        'internet_users': 20.0, 'inflation': 8.0,
        'life_expectancy': 60.0, 'urban_population': 35.0,
    }
    print(predict_poverty(example, 'Sub-Saharan Africa'))