import numpy as np


class MatMul:
    def __init__(self, name, layers_dict):
        self.params = layers_dict[name]
        self.name = name
        self.type = self.params["op"]

        self.transpose_a = self.params["transpose_a"]
        self.transpose_b = self.params["transpose_b"]

        # shape为4维tensor，各参数含义：[width, height, channels, kernel_nums]
        self.w = layers_dict[self.params["input"][1][:-5]]["tensor_content"]
        print(self.w.shape)

    def forward(self, inputs):
        print("inputs: ", inputs.shape)
        print("self.w: ", self.w.shape)
        print("self.transpose_a: ",self.transpose_a)
        print("self.transpose_b: ",self.transpose_b)
        if not self.transpose_a:
            inputs = np.transpose(inputs)
        if self.transpose_b:
            self.w = np.transpose(self.w)

        output = np.matmul(self.w, inputs)
        print(output)
        print("output.shape: ", output.shape)

        return output
