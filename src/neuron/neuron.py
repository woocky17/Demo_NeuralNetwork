import numpy as np  # * Importa la librería NumPy para operaciones numéricas


class Neuron:  # * Define la clase Neuron que representa una neurona artificial
    def __init__(self, n_inputs):  # * Inicializa la neurona con un número de entradas
        """
        Inicializa una neurona artificial con un número específico de entradas.

        Args:
            n_inputs (int): Número de entradas que recibirá la neurona. Determina la cantidad
                          de pesos sinápticos que se crearán.

        Atributos inicializados:
            weights: Array de pesos aleatorios, uno por cada entrada
            bias: Término de sesgo aleatorio que se suma a la combinación lineal
            output: Valor de salida de la neurona (inicialmente 0)
            inputs: Almacena las últimas entradas procesadas
            dweight: Gradientes de los pesos para el entrenamiento
            dbias: Gradiente del sesgo para el entrenamiento
        """
        # * Genera un array de pesos aleatorios siguiendo una distribución normal (media=0, desv=1)
        # * Los pesos son los parámetros que la neurona ajustará durante el entrenamiento
        # * Pesos sinápticos inicializados aleatoriamente
        self.weights = np.random.randn(n_inputs)

        # * Genera un valor aleatorio para el sesgo (bias) siguiendo una distribución normal
        # * El sesgo permite a la neurona ajustar su umbral de activación
        self.bias = np.random.randn()  # * Sesgo inicializado aleatoriamente

        # * Inicializa la salida de la neurona a 0
        # * Este valor se actualizará cada vez que se realice una propagación hacia adelante
        self.output = 0  # * Salida de la neurona (inicialmente 0)

        # * Variable para almacenar las entradas de la última propagación hacia adelante
        # * Necesario para calcular los gradientes durante la retropropagación
        self.inputs = None  # * Últimas entradas procesadas

        # * Inicializa el array de gradientes de los pesos a ceros
        # * Tendrá la misma forma que el array de pesos
        self.dweight = np.zeros_like(self.weights)  # * Gradientes de los pesos

        # * Inicializa el gradiente del sesgo a 0
        # * Se utilizará para actualizar el sesgo durante el entrenamiento
        self.dbias = 0  # * Gradiente del sesgo

    def activate(self, weighted_sum):  # * Aplica la función de activación sigmoide
        """
        Aplica la función de activación sigmoide a la suma ponderada de las entradas.

        Args:
            weighted_sum (float): Suma ponderada de las entradas más el sesgo (z = wx + b)

        Returns:
            float: Valor entre 0 y 1, resultado de aplicar la función sigmoide 1/(1 + e^(-z))
        """
        # * Aplica la función de activación sigmoide: f(x) = 1 / (1 + e^(-x))
        # * Comprime cualquier número real a un valor entre 0 y 1
        # * Esta no-linealidad permite a la red aprender patrones complejos
        # * Devuelve el resultado de la sigmoide
        return 1 / (1 + np.exp(-weighted_sum))

    def derivate_activate(self, sigmoid_output):  # * Derivada de la función sigmoide
        """
        Calcula la derivada de la función de activación sigmoide.

        Args:
            sigmoid_output (float): Valor de salida de la función sigmoide

        Returns:
            float: Derivada de la función sigmoide: sigmoid(x) * (1 - sigmoid(x))
                 Usado en el proceso de retropropagación para actualizar pesos
        """
        # * Calcula la derivada de la función sigmoide: f'(x) = f(x) * (1 - f(x))
        # * Esta derivada se usa en la retropropagación para determinar cómo ajustar los pesos
        # * Es eficiente porque podemos calcularla directamente del valor de salida
        return sigmoid_output * (1 - sigmoid_output)

    def forward(self, inputs):
        """
        Realiza la propagación hacia adelante, calculando la salida de la neurona.

        Args:
            inputs (numpy.array): Array con los valores de entrada a la neurona

        Returns:
            float: Valor de salida de la neurona después de aplicar la función de activación
                 al producto escalar de los pesos y las entradas más el sesgo
        """
        # * Guarda las entradas para usarlas posteriormente en la retropropagación
        self.inputs = inputs

        # * Calcula la suma ponderada: z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
        # * np.dot realiza el producto escalar entre los pesos y las entradas
        weight_sum = np.dot(self.weights, inputs) + self.bias

        # * Aplica la función de activación a la suma ponderada
        # * y guarda el resultado como la salida de la neurona
        self.output = self.activate(weight_sum)

        # * Devuelve el valor de salida calculado
        return self.output

    def backward(self, d_output, learning_rate):
        """
        Realiza la retropropagación del error y actualiza los pesos y el sesgo.

        Args:
            d_output (float): Gradiente del error con respecto a la salida de la neurona
            learning_rate (float): Tasa de aprendizaje que controla el tamaño de los pasos
                                en la actualización de los pesos

        Returns:
            numpy.array: Gradiente del error con respecto a las entradas, necesario para
                       la retropropagación en capas anteriores

        Efectos:
            - Actualiza los pesos (weights) usando el gradiente y la tasa de aprendizaje
            - Actualiza el sesgo (bias) usando el gradiente y la tasa de aprendizaje
        """
        # * Calcula el gradiente local multiplicando el gradiente recibido
        # * por la derivada de la función de activación en el punto actual
        d_activation = d_output * self.derivate_activate(self.output)

        # * Calcula el gradiente para cada peso multiplicando el gradiente local
        # * por la entrada correspondiente (regla de la cadena)
        self.dweight = np.dot(d_activation, self.inputs)

        # * El gradiente del sesgo es igual al gradiente local
        # * porque el sesgo siempre tiene una entrada de 1
        self.dbias = d_activation

        # * Calcula el gradiente con respecto a las entradas
        # * Necesario para propagar el error a las capas anteriores
        d_input = np.dot(self.weights, d_activation)

        # * Actualiza los pesos restando el gradiente multiplicado por la tasa de aprendizaje
        # * w = w - α * ∂E/∂w
        self.weights -= learning_rate * self.dweight

        # * Actualiza el sesgo de manera similar a los pesos
        # * b = b - α * ∂E/∂b
        self.bias -= learning_rate * self.dbias

        # * Devuelve el gradiente con respecto a las entradas
        return d_input


if __name__ == "__main__":
    neuron = Neuron(3)
    inputs = np.array([1, 2, 3])
    output = neuron.forward(inputs)

    print("Output:", output)
