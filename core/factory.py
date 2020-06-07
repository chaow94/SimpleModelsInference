from .layers import *

support_ops = {
    "Conv2D": Conv2D,
    "Relu6": Relu6,
    "Relu": Relu,
    "DepthwiseConv2dNative": DepthwiseConv2dNative,
    "FusedBatchNorm": FusedBatchNorm,
    "AvgPool": AvgPool,
    "MaxPool": MaxPool,
    "BiasAdd": BiasAdd,
    "AddN": AddN,
    "Add": Add,
    "Identity": Identity,
    "MatMul": MatMul,
    "Mul": Mul,
    "Reshape": Reshape,
    "SoftMax": SoftMax,
    "Softmax": Softmax,  # lower or up case
    "Pad": Pad,
    "Sub": Sub,
}


class OpFactory:
    def __init__(self, op_type):
        self.op_type = op_type

    def _create_op(self, name, layers_dict):
        return support_ops[self.op_type](name, layers_dict)
