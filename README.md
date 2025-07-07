# Predicción de AQI con Red Neuronal y Datos de Contaminación

Este proyecto predice el Índice de Calidad del Aire (AQI) usando una red neuronal entrenada con datos reales de contaminación y meteorología. Permite estimar el AQI para los próximos días y exportar los resultados a CSV.

## Estructura del proyecto

- **src/**: Código fuente principal
  - **data/**: Preprocesado de datos
  - **model/**: Entrenamiento, predicción y modelos auxiliares
  - **utils/**: Utilidades
- **config/**: Configuración y constantes
- **data/**: Datasets de entrada y salida

## Uso rápido

1. **Crear entorno virtual:**
   ```sh
   python -m venv .venv
   ```
2. **Activar entorno virtual:**
   ```sh
   .venv\Scripts\activate
   ```
3. **Instalar dependencias:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Ejecutar el script principal:**
   ```sh
   python main.py
   ```

## ¿Qué hace el proyecto?

- Carga datos históricos de contaminación y meteorología desde `data/air_pollution_sample.csv`.
- Preprocesa y normaliza los datos.
- Entrena (o carga) una red neuronal para predecir el AQI.
- Predice el AQI para los próximos 10 días usando una regresión lineal para estimar la evolución de la polución.
- Exporta las predicciones a `predicciones_aqi_10dias.csv`.
- Muestra la categoría de calidad del aire y mensajes interpretativos.

## Personalización

- Puedes modificar el dataset de entrada en `data/air_pollution_sample.csv`.
- El modelo de predicción de polución se encuentra en `src/model/predict_pollution.py`.
- El script principal es `main.py`.

## Requisitos principales

- Python 3.8+
- pandas
- numpy
- scikit-learn
- loguru

Instala todo con:
```sh
pip install -r requirements.txt
```

## Notas

- Si quieres mejorar la predicción de polución, puedes implementar modelos más avanzados en `src/model/predict_pollution.py`.
- Si tienes datos reales de AQI, puedes comparar los resultados y ajustar el modelo.

---
Autor: DOS
