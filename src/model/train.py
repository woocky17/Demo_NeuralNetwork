
import joblib  # * Para guardar y cargar modelos y normalizadores
# * Importa la clase de la red neuronal
from src.neuronal_network.neuronal_network import NeuralNetwork
import os  # * Para comprobar la existencia de archivos


def train_and_save_model(X_train, y_train, num_neurons, input_size, modelo_path, scalerX_path, scalerY_path, scaler_X, scaler_y, epochs=10000, learning_rate=0.1):
    nn = NeuralNetwork()  # * Crea una instancia de la red neuronal
    # * Añade la primera capa
    nn.add_layer(num_neurons=num_neurons, input_size=input_size)
    # * Añade la capa oculta
    nn.add_layer(num_neurons=num_neurons, input_size=input_size)
    # * Añade la capa de salida
    nn.add_layer(num_neurons=1, input_size=input_size)
    nn.train(X_train, y_train, epochs=epochs,
             learning_rate=learning_rate)  # * Entrena la red
    joblib.dump(nn, modelo_path)  # * Guarda el modelo entrenado
    joblib.dump(scaler_X, scalerX_path)  # * Guarda el normalizador de X
    joblib.dump(scaler_y, scalerY_path)  # * Guarda el normalizador de y
    return nn  # * Devuelve la red entrenada


def load_model(modelo_path, scalerX_path, scalerY_path):
    nn = joblib.load(modelo_path)  # * Carga el modelo entrenado
    scaler_X = joblib.load(scalerX_path)  # * Carga el normalizador de X
    scaler_y = joblib.load(scalerY_path)  # * Carga el normalizador de y
    return nn, scaler_X, scaler_y  # * Devuelve el modelo y los normalizadores


def model_exists(modelo_path, scalerX_path, scalerY_path):
    # * Comprueba si existen los archivos necesarios
    return os.path.exists(modelo_path) and os.path.exists(scalerX_path) and os.path.exists(scalerY_path)
