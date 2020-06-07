import numpy as np


class BiasAdd:
    def __init__(self, name, layers_dict):
        self.params = layers_dict[name]
        self.name = name
        self.type = self.params["op"]
        # shape: [width, height, channels, kernel_nums]
        self.bias = layers_dict[self.params["input"][1][:-5]]["tensor_content"]

    def forward(self, X):
        return X + self.bias


class AddN:
    def __init__(self, name, layers_dict):
        self.params = layers_dict[name]
        self.name = name
        self.type = self.params["op"]

    def forward(self, X, Y):
        return X + Y


class Add:
    def __init__(self, name, layers_dict):
        self.params = layers_dict[name]
        self.name = name
        self.type = self.params["op"]
        self.bias = layers_dict[self.params["input"][1][:-5]]["tensor_content"]

    def forward(self, X):
        print(X.shape)
        print("self.bias.shape: ", self.bias.shape)
        self.bias = np.array(self.bias)
        print(self.bias.shape)
        if len(self.bias.shape) < 2:
            self.bias = np.reshape(self.bias, (self.bias.shape[0], 1))

        return X + self.bias


class Reshape:
    def __init__(self, name, layers_dict):
        self.params = layers_dict[name]
        self.name = name
        self.type = self.params["op"]
        self.shape = layers_dict[self.params["input"][1]]["tensor_content"]

    def forward(self, X):
        return np.reshape(X, self.shape)


class Mul:
    def __init__(self, name, layers_dict):
        self.params = layers_dict[name]
        self.name = name
        self.type = self.params["op"]
        # shape: [width, height, channels, kernel_nums]
        self.mul = layers_dict[self.params["input"][1]]["tensor_content"]

    def forward(self, X):
        return X * self.mul


class Sub:
    def __init__(self, name, layers_dict):
        self.params = layers_dict[name]
        self.name = name
        self.type = self.params["op"]
        self.sub = layers_dict[self.params["input"][1]]["tensor_content"]

    def forward(self, X):
        return X - self.sub
