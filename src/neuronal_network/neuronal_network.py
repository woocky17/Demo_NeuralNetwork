import numpy as np  # * Importa la librería NumPy para operaciones numéricas
# * Importa la clase Layer para construir las capas de la red
from src.layer.layer import Layer


class NeuralNetwork:  # * Define la clase principal de la red neuronal
    def __init__(self):  # * Inicializa la red neuronal
        """
        Inicializa una red neuronal con una lista de capas.

        Args:
            layers (list): Lista de instancias de la clase Layer que componen la red
            loss_list (list): Lista para almacenar las funciones de pérdida utilizadas en el entrenamiento
        """
        self.layers = []  # * Lista que almacenará las capas de la red
        # * Lista para guardar el historial de pérdidas (loss) durante el entrenamiento
        self.loss_list = []

    def add_layer(self, num_neurons, input_size):  # * Añade una nueva capa a la red
        """
        Agrega una nueva capa a la red neuronal.
        Si es la primera capa, se inicializa con el tamaño de entrada especificado.
        Si no es la primera capa, se utiliza el tamaño de salida de la capa anterior como tamaño de entrada.
        Args:
            num_neurons (int): Número de neuronas en la nueva capa
            input_size (int): Número de entradas que cada neurona recibirá
        """
        if not self.layers:  # * Si es la primera capa...
            # * ...usa el tamaño de entrada dado
            self.layers.append(Layer(num_neurons, input_size))
        else:
            # * Obtiene el tamaño de salida de la última capa
            previous_layer_size = len(self.layers[-1].neurons)
            # * Usa ese tamaño como input_size
            self.layers.append(Layer(num_neurons, previous_layer_size))

    def forward(self, inputs):  # * Propagación hacia adelante (forward pass)
        """
        Realiza la propagación hacia adelante a través de todas las capas de la red.
        Args:
            inputs (numpy.array): Array con los valores de entrada a la red
        Returns:
            numpy.array: Array con la salida final de la red después de pasar por todas las capas
        """
        for layer in self.layers:  # * Pasa los datos por cada capa
            # * Calcula la salida de la capa actual
            inputs = layer.forward(inputs)
        return inputs  # * Devuelve la salida final

    # * Retropropagación del error (backward pass)
    def backward(self, loss_gradient, learning_rate):
        """
        Realiza la retropropagación del error a través de todas las capas de la red
        y actualiza los pesos de las neuronas.
        Args:
            loss_gradient (numpy.array): Gradiente del error con respecto a la salida de la red
            learning_rate (float): Tasa de aprendizaje que controla el tamaño de los pasos
                                  en la actualización de los pesos
        """
        for layer in reversed(self.layers):  # * Recorre las capas en orden inverso
            # * Actualiza los pesos de la capa
            loss_gradient = layer.backward(loss_gradient, learning_rate)

    def train(self, X, y, epochs=1000, learning_rate=0.01):  # * Entrena la red neuronal
        """
        Entrena la red neuronal utilizando el algoritmo de retropropagación.
        Args:
            X (numpy.array): Array de entradas de entrenamiento
            y (numpy.array): Array de salidas esperadas para las entradas de entrenamiento
            epochs (int): Número de épocas para entrenar la red
            learning_rate (float): Tasa de aprendizaje que controla el tamaño de los pasos
                                  en la actualización de los pesos
        """
        for epoch in range(epochs):  # * Itera sobre el número de épocas
            loss = 0  # * Inicializa la pérdida de la época
            for i in range(len(X)):  # * Itera sobre cada muestra de entrenamiento
                # * Calcula la salida de la red para la muestra
                output = self.forward(X[i])
                # * Calcula el error cuadrático medio (MSE)
                loss += np.mean((output - y[i]) ** 2)
                # * Calcula el gradiente de la pérdida
                loss_gradient = 2 * (output - y[i])
                # * Actualiza los pesos usando retropropagación
                self.backward(loss_gradient, learning_rate)
            loss /= len(X)  # * Calcula la pérdida media de la época
            self.loss_list.append(loss)  # * Guarda la pérdida en la lista
            if (epoch + 1) % 100 == 0:  # * Muestra el progreso cada 100 épocas
                # * Imprime la pérdida actual
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    def predict(self, X):  # * Realiza predicciones con la red entrenada
        """
        Realiza predicciones utilizando la red neuronal entrenada.
        Args:
            X (numpy.array): Array de entradas para las cuales se desea realizar predicciones
        Returns:
            numpy.array: Array con las salidas predichas por la red
        """
        predictions = []  # * Lista para almacenar las predicciones
        for i in range(len(X)):  # * Itera sobre cada muestra de entrada
            # * Calcula la salida de la red para la muestra
            output = self.forward(X[i])
            predictions.append(output)  # * Añade la predicción a la lista
        return np.array(predictions)  # * Devuelve las predicciones como array
