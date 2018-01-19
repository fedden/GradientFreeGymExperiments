import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class NeuralNetwork():


    def __init__(self,
                 input_size,
                 hidden_sizes,
                 output_size,
                 output_activation,
                 discrete_action):
        last_idx = len(hidden_sizes) - 1
        self.shapes = [(input_size, hidden_sizes[0])]
        for i in range(len(hidden_sizes) - 1):
            self.shapes.append((hidden_sizes[i], hidden_sizes[i+1]))
        self.shapes.append((hidden_sizes[last_idx], output_size))

        self.weights = []
        self.biases = []
        self.activations = [np.tanh for _ in range(len(hidden_sizes))]
        self.activations.append(output_activation)

        for shape in self.shapes:
            self.weights.append(np.random.uniform(size=shape))
            self.biases.append(np.ones(shape=shape[1]))

        self.discrete_action = discrete_action


    def forward(self,
                inputs):
        output = np.array(inputs).flatten()

        amount_layers = len(self.weights)
        for i in range(amount_layers):
            weights = self.weights[i]
            bias = self.biases[i]
            output = np.matmul(output, weights) + bias

            if self.activations[i] is not None:
                output = self.activations[i](output)

        if self.discrete_action:
            output = np.argmax(np.random.multinomial(1, output))

        return output


    def get_weights(self):
        weights_1d = []
        for i in range(len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
            weights_1d += w.flatten().tolist()
            weights_1d += b.flatten().tolist()
        return np.array(weights_1d)


    def set_weights(self, weights):
        pointer = 0
        for i, weights_shape in enumerate(self.shapes):
            bias_shape = weights_shape[1]
            size = np.prod(weights_shape) + bias_shape
            chunk = np.array(weights[pointer:pointer+size])
            self.weights[i] = chunk[:np.prod(weights_shape)].reshape(weights_shape)
            self.biases[i] = chunk[np.prod(weights_shape):].reshape(bias_shape)
            pointer += size
