import numpy as np


class Relu:
    def __init__(self, *args):
        pass

    def forward(self, inputs):
        return np.where(inputs >= 0, inputs, 0)


class Relu6:
    def __init__(self, *args):
        pass

    def forward(self, inputs):
        # min(max(features, 0), 6)
        return np.minimum(np.maximum(0, inputs), 6)


class SoftMax:
    def __init__(self, *args):
        pass

    def forward(self, inputs):
        inputs_ = inputs - inputs.max()
        e = np.exp(inputs_)
        return e / np.sum(e, axis=0, keepdims=True)

class Softmax:
    def __init__(self, *args):
        pass

    def forward(self, inputs):
        inputs_ = inputs - inputs.max()
        e = np.exp(inputs_)
        return e / np.sum(e, axis=0, keepdims=True)
