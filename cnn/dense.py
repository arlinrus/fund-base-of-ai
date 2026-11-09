import numpy as np

class Dense:
    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) * np.sqrt(2 / input_len)
        self.biases = np.zeros(nodes)

    def forward(self, input):
        self.last_input_shape = input.shape
        self.last_input = input.flatten()

        totals = self.last_input @ self.weights + self.biases
        self.last_relu_mask = totals > 0
        self.last_output = totals * self.last_relu_mask
        return self.last_output

    def backprob(self, dl_dout, lr):
        # градиент через ReLU
        dl_dt = dl_dout * self.last_relu_mask

        dl_dw = np.outer(self.last_input, dl_dt)
        dl_dinput = self.weights @ dl_dt

        self.weights -= lr * dl_dw
        self.biases -= lr * dl_dt

        return dl_dinput.reshape(self.last_input_shape)