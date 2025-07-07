from src.model.predict_pollution import predict_next_pollution

import sys

from loguru import logger
import sys
import pandas as pd
import numpy as np
from src.data.preprocessing import load_and_prepare_data
from src.model.train import train_and_save_model, load_model, model_exists
from src.model.predict import predict_and_inverse, predict_next_day
import os


def main():

    # * Función para categorizar el AQI

    def aqi_category(aqi):
        if aqi <= 19:
            return "Idónea"
        elif aqi <= 49:
            return "Buena"
        elif aqi <= 99:
            return "Mala"
        elif aqi <= 149:
            return "Poco saludable"
        elif aqi <= 249:
            return "Muy poco saludable"
        else:
            return "Peligrosa"

    logger.remove()
    logger.add(sys.stderr, colorize=True, level='INFO',
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <magenta>{module}</magenta> | <cyan>{function}:{line}</cyan> | <level>{message}</level>")
    logger.add(".log", rotation='30 days', retention=12, colorize=False, level='INFO',
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module} | {function}:{line} | {message}")

    logger.info('Process has started')

    # * Definir variables antes de usarlas
    features = ['PM2.5', 'PM10', 'NO2', 'O3', 'SO2',
                'CO', 'temperature', 'humidity', 'wind_speed']
    target = 'target'
    modelo_path = 'modelo_entrenado.pkl'
    scalerX_path = 'scalerX.pkl'
    scalerY_path = 'scalerY.pkl'

    # * Cargar datos reales de contaminación del aire desde un archivo CSV
    X_train, X_test, y_train, y_test, scaler_X, scaler_y, df = load_and_prepare_data(
        'data/air_pollution_sample.csv', features, target)
    num_neurons = X_train.shape[1]
    input_size = X_train.shape[1]

    if model_exists(modelo_path, scalerX_path, scalerY_path):
        nn, scaler_X, scaler_y = load_model(
            modelo_path, scalerX_path, scalerY_path)
        logger.info('Modelo y normalizadores cargados desde disco.')
    else:
        nn = train_and_save_model(X_train, y_train, num_neurons, input_size,
                                  modelo_path, scalerX_path, scalerY_path, scaler_X, scaler_y)
        logger.info('Modelo y normalizadores guardados en disco.')

    # * Predicción autoregresiva del AQI para los próximos 10 días usando polución estimada
    predicciones_aqi = []
    fechas_pred = []
    fecha_base = pd.to_datetime(df['date'].max())
    # Usamos la función predict_next_pollution para estimar la polución de cada día futuro
    polucion_futura = predict_next_pollution(df, days=10)
    for i in range(10):
        X_input = np.array(polucion_futura[i]).reshape(1, -1)
        X_input_scaled = scaler_X.transform(X_input)
        pred_aqi_scaled = nn.predict(X_input_scaled)
        pred_aqi = scaler_y.inverse_transform(pred_aqi_scaled)[0, 0]
        predicciones_aqi.append(pred_aqi)
        fechas_pred.append(
            (fecha_base + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d'))
    logger.info(
        'Predicciones de AQI para los próximos 10 días usando polución estimada:')
    categorias_pred = [aqi_category(pred) for pred in predicciones_aqi]
    for fecha, pred, cat in zip(fechas_pred, predicciones_aqi, categorias_pred):
        logger.info(f"{fecha}: {pred:.2f} AQI - {cat}")

    # Guardar predicciones en un CSV
    df_predicciones = pd.DataFrame({
        'fecha': fechas_pred,
        'aqi_predicho': predicciones_aqi,
        'categoria': categorias_pred
    })
    df_predicciones.to_csv('predicciones_aqi_10dias.csv', index=False)
    logger.info('Predicciones guardadas en predicciones_aqi_10dias.csv')

    predictions, y_test_original = predict_and_inverse(
        nn, scaler_y, X_test, y_test)

    logger.info('Predicciones reales y su categoría:')
    for pred in predictions.flatten():
        logger.info(f"{pred:.2f} AQI - {aqi_category(pred)}")
    logger.info('Valores reales:')
    logger.info(f'{y_test_original.flatten()}')

    logger.success('Process finished successfully')


if __name__ == "__main__":
    main()
