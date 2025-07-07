import numpy as np
from src.layer.layer import Layer


class NeuralNetwork:
    def __init__(self):
        """
        Inicializa una red neuronal con una lista de capas.

        Args:
            layers (list): Lista de instancias de la clase Layer que componen la red
            loss_list (list): Lista para almacenar las funciones de pérdida utilizadas en el entrenamiento
        """
        self.layers = []
        self.loss_list = []

    def add_layer(self, num_neurons, input_size):
        """
        Agrega una nueva capa a la red neuronal.
        Si es la primera capa, se inicializa con el tamaño de entrada especificado.
        Si no es la primera capa, se utiliza el tamaño de salida de la capa anterior como tamaño de entrada.
        Args:
            num_neurons (int): Número de neuronas en la nueva capa
            input_size (int): Número de entradas que cada neurona recibirá
        """
        if not self.layers:
            self.layers.append(Layer(num_neurons, input_size))
        else:
            previous_layer_size = len(self.layers[-1].neurons)
            self.layers.append(Layer(num_neurons, previous_layer_size))

    def forward(self, inputs):
        """
        Realiza la propagación hacia adelante a través de todas las capas de la red.
        Args:
            inputs (numpy.array): Array con los valores de entrada a la red
        Returns:
            numpy.array: Array con la salida final de la red después de pasar por todas las capas
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, loss_gradient, learning_rate):
        """
        Realiza la retropropagación del error a través de todas las capas de la red
        y actualiza los pesos de las neuronas.
        Args:
            loss_gradient (numpy.array): Gradiente del error con respecto a la salida de la red
            learning_rate (float): Tasa de aprendizaje que controla el tamaño de los pasos
                                  en la actualización de los pesos
        """
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient, learning_rate)

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        """
        Entrena la red neuronal utilizando el algoritmo de retropropagación.
        Args:
            X (numpy.array): Array de entradas de entrenamiento
            y (numpy.array): Array de salidas esperadas para las entradas de entrenamiento
            epochs (int): Número de épocas para entrenar la red
            learning_rate (float): Tasa de aprendizaje que controla el tamaño de los pasos
                                  en la actualización de los pesos
        """
        for epoch in range(epochs):
            loss = 0
            for i in range(len(X)):
                output = self.forward(X[i])
                loss += np.mean((output - y[i]) ** 2)  # MSE loss
                loss_gradient = 2 * (output - y[i])
                self.backward(loss_gradient, learning_rate)
            loss /= len(X)
            self.loss_list.append(loss)
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        """
        Realiza predicciones utilizando la red neuronal entrenada.
        Args:
            X (numpy.array): Array de entradas para las cuales se desea realizar predicciones
        Returns:
            numpy.array: Array con las salidas predichas por la red
        """
        predictions = []
        for i in range(len(X)):
            output = self.forward(X[i])
            predictions.append(output)
        return np.array(predictions)
