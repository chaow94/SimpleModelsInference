import numpy as np


class FusedBatchNorm:

    def __init__(self, name, layers_dict):
        self.params = layers_dict[name]
        self.name = name
        self.type = self.params["op"]
        self.epsilon = self.params["epsilon"]

        self.gamma = layers_dict[self.params["input"][1][:-5]]["tensor_content"]
        self.beta = layers_dict[self.params["input"][2][:-5]]["tensor_content"]
        self.moving_mean = layers_dict[self.params["input"][3][:-5]]["tensor_content"]
        self.moving_variance = layers_dict[self.params["input"][4][:-5]]["tensor_content"]

    def forward(self, inputs):
        var_sqrt = np.sqrt(self.moving_variance + self.epsilon)
        y = self.gamma * ((inputs - self.moving_mean) / var_sqrt) + self.beta

        return y
