import joblib
import numpy as np

model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

def predict_poverty(input_dict):
    """
    input_dict: {feature_name: value} for all model features.
    Returns predicted poverty ratio (% below $3.00/day).
    """
    # Arrange inputs in the exact training order
    ordered = [input_dict[name] for name in feature_names]
    X_new = np.array(ordered).reshape(1, -1)

    # Scale using the SAME scaler from training
    X_scaled = scaler.transform(X_new)

    prediction = model.predict(X_scaled)[0]
    return round(float(prediction), 2)


if __name__ == '__main__':
    # Quick test with one example
    sample = {name: 0 for name in feature_names}  # replace with real values
    print('Predicted poverty ratio:', predict_poverty(sample))