import numpy as np


class Linear:
    def __init__(self, input_chn, output_chn, weights, non_linear=None):
        self.input_chn = input_chn
        self.output_chn = output_chn
        self.non_linear = non_linear
        self.weights = weights
        self.top_layer = []

    def forward(self, inputs):
        output = np.dot(self.weights, inputs)
        if self.non_linear:
            output = self.non_linear(output)
        return output
