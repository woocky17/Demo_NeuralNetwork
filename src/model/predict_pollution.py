import pandas as pd
import numpy as np


from sklearn.linear_model import LinearRegression


def predict_next_pollution(df_hist, days=10):
    """
    Predice los próximos valores de polución usando regresión lineal para cada feature.
    Devuelve un array (days, n_features) con los valores de features para cada día futuro.
    """
    cols = ['PM2.5', 'PM10', 'NO2', 'O3', 'SO2',
            'CO', 'temperature', 'humidity', 'wind_speed']
    n = len(df_hist)
    X = np.arange(n).reshape(-1, 1)
    preds = []
    for feat in cols:
        y = df_hist[feat].values
        model = LinearRegression()
        model.fit(X, y)
        X_future = np.arange(n, n+days).reshape(-1, 1)
        y_future = model.predict(X_future)
        preds.append(y_future)
    preds = np.array(preds).T  # shape (days, n_features)
    return preds

# Ejemplo de uso:
# df_hist = pd.read_csv('data/air_pollution_sample.csv')
# pred = predict_next_pollution(df_hist, days=10)
# print(pred)
