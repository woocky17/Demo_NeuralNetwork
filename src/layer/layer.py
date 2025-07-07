import numpy as np
from src.neuron.neuron import Neuron


class Layer:

    def __init__(self, num_neurons, input_size):
        """
        Inicializa una capa de neuronas.

        Args:
            num_neurons (int): Número de neuronas en la capa
            input_size (int): Número de entradas que cada neurona recibirá
        """
        self.neurons = [Neuron(input_size) for _ in range(num_neurons)]

    def forward(self, inputs):
        """
        Realiza la propagación hacia adelante a través de la capa.
        Args:
            inputs (numpy.array): Array con los valores de entrada a la capa
        Returns:
            numpy.array: Array con las salidas de cada neurona en la capa
        """
        return np.array([Neuron.forward(inputs) for Neuron in self.neurons])

    def backward(self, d_output, learning_rate):
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
        d_inputs = np.zeros(len(self.neurons[0].inputs))
        for i, neuron in enumerate(self.neurons):
            d_inputs += neuron.backward(d_output[i], learning_rate)
        return d_inputs
