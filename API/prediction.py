import joblib
import numpy as np

model = joblib.load('API/best_model.pkl')
scaler = joblib.load('API/scaler.pkl')
feature_names = joblib.load('API/feature_names.pkl')

REGION_COLUMNS = [f for f in feature_names if f.startswith('reg_')]

# Maps a user-facing region name to the dummy column it activates.
# The baseline region (dropped by drop_first=True) maps to None = all zeros.
REGION_MAP = {
    'Europe & Central Asia': 'reg_europe_centralasia',
    'Latin America & Caribbean': 'reg_latam_caribbean',
    'Middle East, North Africa, Afghanistan & Pakistan': 'reg_mena',
    'North America': 'reg_northamerica',
    'South Asia': 'reg_southasia',
    'Sub-Saharan Africa': 'reg_subsaharan',
    'East Asia & Pacific': None,   # baseline: all region dummies = 0
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