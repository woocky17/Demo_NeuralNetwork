import joblib
from src.neuronal_network.neuronal_network import NeuralNetwork
import os


def train_and_save_model(X_train, y_train, num_neurons, input_size, modelo_path, scalerX_path, scalerY_path, scaler_X, scaler_y, epochs=10000, learning_rate=0.1):
    nn = NeuralNetwork()
    nn.add_layer(num_neurons=num_neurons, input_size=input_size)
    nn.add_layer(num_neurons=num_neurons, input_size=input_size)
    nn.add_layer(num_neurons=1, input_size=input_size)
    nn.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate)
    joblib.dump(nn, modelo_path)
    joblib.dump(scaler_X, scalerX_path)
    joblib.dump(scaler_y, scalerY_path)
    return nn


def load_model(modelo_path, scalerX_path, scalerY_path):
    nn = joblib.load(modelo_path)
    scaler_X = joblib.load(scalerX_path)
    scaler_y = joblib.load(scalerY_path)
    return nn, scaler_X, scaler_y


def model_exists(modelo_path, scalerX_path, scalerY_path):
    return os.path.exists(modelo_path) and os.path.exists(scalerX_path) and os.path.exists(scalerY_path)
