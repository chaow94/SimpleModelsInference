import numpy as np


class Pad:

    def __init__(self, name, layers_dict):
        self.params = layers_dict[name]
        self.name = name
        self.type = self.params["op"]

        self.paddings = layers_dict[self.params["input"][1]]["tensor_content"]
        print(self.paddings)

    def forward(self, inputs):
        return np.pad(inputs, (self.paddings[0], self.paddings[1],
                               self.paddings[2], self.paddings[3]), mode='constant')

# def pad_inputs(X, pad):
#     return np.pad(X, ((0, 0), (int(pad[0] // 2), pad[0] - int(pad[0] // 2)),
#                       (int(pad[1] // 2), pad[1] - int(pad[1] // 2)), (0, 0)), mode='constant')
