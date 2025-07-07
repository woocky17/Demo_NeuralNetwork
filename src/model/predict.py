import numpy as np


def predict_and_inverse(nn, scaler_y, X_test, y_test):
    predictions_scaled = nn.predict(X_test)
    predictions = scaler_y.inverse_transform(predictions_scaled)
    y_test_original = scaler_y.inverse_transform(y_test)
    return predictions, y_test_original


def predict_next_day(nn, scaler_X, scaler_y, siguiente_dia):
    siguiente_dia_scaled = scaler_X.transform(siguiente_dia)
    prediccion_siguiente_scaled = nn.predict(siguiente_dia_scaled)
    prediccion_siguiente = scaler_y.inverse_transform(
        prediccion_siguiente_scaled)
    return prediccion_siguiente
