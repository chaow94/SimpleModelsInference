import numpy as np


class AvgPool:
    def __init__(self, name, layers_dict):

        self.params = layers_dict[name]
        self.name = name
        self.type = self.params["op"]
        self.ksize = self.params["ksize"]
        self.strides = self.params["strides"]

    def forward(self, inputs):

        batch_size, h, w, channel = inputs.shape
        _, k_h, k_w, _ = self.params['ksize']

        out_h = int(1 + (h - k_h) / self.params['strides'][1])
        out_w = int(1 + (w - k_w) / self.params['strides'][2])

        res = np.zeros((batch_size, out_h, out_w, channel))

        for i in range(batch_size):
            for h in range(out_h):
                for w in range(out_w):

                    x = h * self.params['strides'][1]
                    y = w * self.params['strides'][2]

                    for c in range(channel):
                        res[i, h, w, c] = np.mean(inputs[i, x: x + k_w, y: y + k_h, c])

        return res


class MaxPool:
    def __init__(self, name, layers_dict):

        self.params = layers_dict[name]
        self.name = name
        self.type = self.params["op"]
        self.ksize = self.params["ksize"]
        self.strides = self.params["strides"]

    def forward(self, inputs):

        batch_size, h, w, channel = inputs.shape
        _, k_h, k_w, _ = self.params['ksize']

        out_h = int(1 + (h - k_h) / self.params['strides'][1])
        out_w = int(1 + (w - k_w) / self.params['strides'][2])

        res = np.zeros((batch_size, out_h, out_w, channel))

        for i in range(batch_size):
            for h in range(out_h):
                for w in range(out_w):

                    x = h * self.params['strides'][1]
                    y = w * self.params['strides'][2]

                    for c in range(channel):
                        res[i, h, w, c] = np.max(inputs[i, x: x + k_w, y: y + k_h, c])

        return res
