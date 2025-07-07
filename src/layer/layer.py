import numpy as np  # * Importa la librería NumPy para operaciones numéricas
# * Importa la clase Neuron para crear las neuronas de la capa
from src.neuron.neuron import Neuron


class Layer:  # * Define la clase Layer que representa una capa de la red

    # * Inicializa la capa con un número de neuronas y tamaño de entrada
    def __init__(self, num_neurons, input_size):
        """
        Inicializa una capa de neuronas.

        Args:
            num_neurons (int): Número de neuronas en la capa
            input_size (int): Número de entradas que cada neurona recibirá
        """
        self.neurons = [Neuron(input_size) for _ in range(
            num_neurons)]  # * Crea la lista de neuronas de la capa

    def forward(self, inputs):  # * Propagación hacia adelante de la capa
        """
        Realiza la propagación hacia adelante a través de la capa.
        Args:
            inputs (numpy.array): Array con los valores de entrada a la capa
        Returns:
            numpy.array: Array con las salidas de cada neurona en la capa
        """
        return np.array([Neuron.forward(inputs) for Neuron in self.neurons])  # * Calcula la salida de cada neurona

    def backward(self, d_output, learning_rate):  # * Retropropagación del error en la capa
        """
        Realiza la retropropagación del error a través de la capa.
        Args:
            d_output (numpy.array): Gradiente del error con respecto a las salidas de la capa
            learning_rate (float): Tasa de aprendizaje que controla el tamaño de los pasos
                                  en la actualización de los pesos
        Returns:
            numpy.array: Gradiente del error con respecto a las entradas de la capa,
                         necesario para propagar el error a las capas anteriores
        """
        d_inputs = np.zeros(
            # * Inicializa el gradiente de entrada
            len(self.neurons[0].inputs))
        # * Itera sobre cada neurona de la capa
        for i, neuron in enumerate(self.neurons):
            # * Acumula el gradiente de entrada
            d_inputs += neuron.backward(d_output[i], learning_rate)
        return d_inputs  # * Devuelve el gradiente total para la capa anterior
