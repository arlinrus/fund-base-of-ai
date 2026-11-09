import numpy as np

# pick the digit with the highest output from the network
# softmax helps us quantify how sure we r of our prediction, which is useful when trining and evaluating CNN
class SoftMax:
    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        self.last_input_shape = input.shape
        self.last_input = input.flatten()

        totals = self.last_input @ self.weights + self.biases
        shifted = totals - np.max(totals)
        self.last_log_probs = (shifted - np.log(np.sum(np.exp(shifted))))

        self.last_output = np.exp(self.last_log_probs)

        return self.last_output

    def backprob(self, label, lr):
        dl_dt = self.last_output.copy()
        dl_dt[label] -= 1

        dl_dw = np.outer(self.last_input, dl_dt)

        dl_dinput = self.weights @ dl_dt

        self.weights -= lr * dl_dw
        self.biases -= lr * dl_dt

        return dl_dinput.reshape(self.last_input_shape)


    



    