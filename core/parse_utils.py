from collections import OrderedDict

import numpy as np
from tensorflow.python.framework import tensor_util


class Para:
    layers_dict = {}


class Identity:
    def __init__(self, layer_para):
        super(Identity, self).__init__()
        self.layer_para = layer_para
        cur_dict = self._parse()
        Para.layers_dict[cur_dict["name"]] = cur_dict

    def _parse(self):
        dic = OrderedDict()
        dic["name"] = str(self.layer_para.name)
        dic["op"] = str(self.layer_para.op)
        dic["input"] = list(self.layer_para.input)

        dic["type"] = self.layer_para.attr["T"].type

        return dic


class Conv2D:

    def __init__(self, layer_para):
        super(Conv2D, self).__init__()
        self.layer_para = layer_para
        cur_dict = self._parse()
        Para.layers_dict[cur_dict["name"]] = cur_dict

    def _parse(self):
        dic = OrderedDict()
        dic["name"] = str(self.layer_para.name)
        dic["op"] = str(self.layer_para.op)
        dic["input"] = list(self.layer_para.input)

        dic["type"] = self.layer_para.attr["T"].type
        dic["data_format"] = self.layer_para.attr["data_format"].s
        dic["dilations"] = self.layer_para.attr["dilations"].list.i
        # TODO
        dic["explicit_paddings"] = self.layer_para.attr["explicit_paddings"].list
        dic["padding"] = bytes.decode(self.layer_para.attr["padding"].s)
        dic["strides"] = list(self.layer_para.attr["strides"].list.i)
        dic["use_cudnn_on_gpu"] = self.layer_para.attr["use_cudnn_on_gpu"].b

        return dic


class DepthwiseConv2dNative:

    def __init__(self, layer_para):
        super(DepthwiseConv2dNative, self).__init__()
        self.layer_para = layer_para
        cur_dict = self._parse()
        Para.layers_dict[cur_dict["name"]] = cur_dict
        # #print(layer_dict)

    def _parse(self):
        dic = OrderedDict()
        dic["name"] = self.layer_para.name
        dic["op"] = self.layer_para.op
        dic["input"] = self.layer_para.input
        dic["type"] = self.layer_para.attr["T"].type
        dic["data_format"] = bytes.decode(self.layer_para.attr["data_format"].s)
        dic["dilations"] = self.layer_para.attr["dilations"].list.i
        # TODO
        dic["explicit_paddings"] = self.layer_para.attr["explicit_paddings"].list
        dic["padding"] = bytes.decode(self.layer_para.attr["padding"].s)
        dic["strides"] = self.layer_para.attr["strides"].list.i
        dic["use_cudnn_on_gpu"] = self.layer_para.attr["use_cudnn_on_gpu"].b

        return dic


class FusedBatchNorm:

    def __init__(self, layer_para):
        super(FusedBatchNorm, self).__init__()
        self.layer_para = layer_para
        cur_dict = self._parse()
        Para.layers_dict[cur_dict["name"]] = cur_dict

    def _parse(self):
        dic = OrderedDict()
        dic["name"] = self.layer_para.name
        dic["op"] = self.layer_para.op
        dic["input"] = self.layer_para.input
        dic["type"] = self.layer_para.attr["T"].type
        dic["data_format"] = self.layer_para.attr["data_format"].s
        dic["epsilon"] = self.layer_para.attr["epsilon"].f

        return dic


class Relu6:

    def __init__(self, layer_para):
        super(Relu6, self).__init__()
        self.layer_para = layer_para
        cur_dict = self._parse()
        Para.layers_dict[cur_dict["name"]] = cur_dict
        # print(layer_dict)

    def _parse(self):
        dic = OrderedDict()
        dic["name"] = self.layer_para.name
        dic["op"] = self.layer_para.op
        dic["input"] = self.layer_para.input
        dic["type"] = self.layer_para.attr["T"].type

        return dic


class Relu:

    def __init__(self, layer_para):
        super(Relu, self).__init__()
        self.layer_para = layer_para
        cur_dict = self._parse()
        Para.layers_dict[cur_dict["name"]] = cur_dict

    def _parse(self):
        dic = OrderedDict()
        dic["name"] = self.layer_para.name
        dic["op"] = self.layer_para.op
        dic["input"] = self.layer_para.input
        dic["type"] = self.layer_para.attr["T"].type

        return dic


class AvgPool:

    def __init__(self, layer_para):
        super(AvgPool, self).__init__()
        self.layer_para = layer_para
        cur_dict = self._parse()
        Para.layers_dict[cur_dict["name"]] = cur_dict

    def _parse(self):
        dic = OrderedDict()
        dic["name"] = self.layer_para.name
        dic["op"] = self.layer_para.op
        dic["input"] = self.layer_para.input
        dic["type"] = self.layer_para.attr["T"].type
        dic["data_format"] = self.layer_para.attr["data_format"].s
        dic["ksize"] = self.layer_para.attr["ksize"].list.i
        dic["padding"] = self.layer_para.attr["padding"].s
        dic["strides"] = self.layer_para.attr["strides"].i

        return dic


class BiasAdd:

    def __init__(self, layer_para):
        super(BiasAdd, self).__init__()
        self.layer_para = layer_para
        cur_dict = self._parse()
        Para.layers_dict[cur_dict["name"]] = cur_dict

    def _parse(self):
        dic = OrderedDict()
        dic["name"] = self.layer_para.name
        dic["op"] = self.layer_para.op
        dic["input"] = self.layer_para.input
        dic["type"] = self.layer_para.attr["T"].type
        dic["data_format"] = self.layer_para.attr["data_format"].s

        return dic


class Add:

    def __init__(self, layer_para):
        super(Add, self).__init__()
        self.layer_para = layer_para
        cur_dict = self._parse()
        Para.layers_dict[cur_dict["name"]] = cur_dict
        # print(layer_dict)

    def _parse(self):
        dic = OrderedDict()
        dic["name"] = self.layer_para.name
        dic["op"] = self.layer_para.op
        dic["input"] = self.layer_para.input
        dic["type"] = self.layer_para.attr["T"].type
        dic["data_format"] = self.layer_para.attr["data_format"].s

        return dic


class AddN:

    def __init__(self, layer_para):
        super(AddN, self).__init__()
        self.layer_para = layer_para
        cur_dict = self._parse()
        Para.layers_dict[cur_dict["name"]] = cur_dict
        # print(layer_dict)

    def _parse(self):
        dic = OrderedDict()
        dic["name"] = self.layer_para.name
        dic["op"] = self.layer_para.op
        dic["input"] = self.layer_para.input
        dic["type"] = self.layer_para.attr["T"].type
        dic["data_format"] = self.layer_para.attr["data_format"].s

        return dic


class AvgPool:

    def __init__(self, layer_para):
        self.layer_para = layer_para
        cur_dict = self._parse()
        Para.layers_dict[cur_dict["name"]] = cur_dict

    def _parse(self):
        dic = OrderedDict()
        dic["name"] = self.layer_para.name
        dic["op"] = self.layer_para.op
        dic["input"] = self.layer_para.input
        dic["type"] = self.layer_para.attr["T"].type
        dic["data_format"] = bytes.decode(self.layer_para.attr["data_format"].s)
        dic["padding"] = bytes.decode(self.layer_para.attr["padding"].s)
        dic["strides"] = self.layer_para.attr["strides"].list.i
        dic["ksize"] = self.layer_para.attr["ksize"].list.i

        return dic


class MaxPool:

    def __init__(self, layer_para):
        self.layer_para = layer_para
        cur_dict = self._parse()
        Para.layers_dict[cur_dict["name"]] = cur_dict

    def _parse(self):
        dic = OrderedDict()
        dic["name"] = self.layer_para.name
        dic["op"] = self.layer_para.op
        dic["input"] = self.layer_para.input
        dic["type"] = self.layer_para.attr["T"].type
        dic["data_format"] = bytes.decode(self.layer_para.attr["data_format"].s)
        dic["padding"] = bytes.decode(self.layer_para.attr["padding"].s)
        dic["strides"] = self.layer_para.attr["strides"].list.i
        dic["ksize"] = self.layer_para.attr["ksize"].list.i

        return dic


class Const:

    def __init__(self, layer_para):
        super(Const, self).__init__()
        self.layer_para = layer_para
        cur_dict = self._parse()
        Para.layers_dict[cur_dict["name"]] = cur_dict

    def _parse(self):
        dic = OrderedDict()
        dic["name"] = str(self.layer_para.name)
        dic["op"] = str(self.layer_para.op)
        dic["input"] = list(self.layer_para.input)
        dic["type"] = str(self.layer_para.attr["T"].type)
        # TODO
        n = len(self.layer_para.attr["value"].tensor.tensor_shape.dim)
        shape = []
        for i in range(n):
            size = str(self.layer_para.attr["value"].tensor.tensor_shape.dim[i])
            size = int(size.split(": ")[1])
            shape.append(size)
        dic["tensor_shape"] = shape

        dic["tensor_content"] = np.reshape(
            tensor_util.MakeNdarray(self.layer_para.attr["value"].tensor), shape)

        return dic


class MatMul:

    def __init__(self, layer_para):
        super(MatMul, self).__init__()
        self.layer_para = layer_para
        cur_dict = self._parse()
        Para.layers_dict[cur_dict["name"]] = cur_dict

    def _parse(self):
        dic = OrderedDict()
        dic["name"] = str(self.layer_para.name)
        dic["op"] = str(self.layer_para.op)
        dic["input"] = list(self.layer_para.input)

        dic["type"] = self.layer_para.attr["T"].type
        dic["data_format"] = self.layer_para.attr["data_format"].s
        dic["transpose_a"] = self.layer_para.attr["transpose_a"].b
        dic["transpose_b"] = self.layer_para.attr["transpose_a"].b

        return dic


class Softmax:

    def __init__(self, layer_para):
        super(Softmax, self).__init__()
        self.layer_para = layer_para
        cur_dict = self._parse()
        Para.layers_dict[cur_dict["name"]] = cur_dict

    def _parse(self):
        dic = OrderedDict()
        dic["name"] = str(self.layer_para.name)
        dic["op"] = str(self.layer_para.op)
        dic["input"] = list(self.layer_para.input)

        dic["type"] = self.layer_para.attr["T"].type
        return dic


class SoftMax:

    def __init__(self, layer_para):
        super(SoftMax, self).__init__()
        self.layer_para = layer_para
        cur_dict = self._parse()
        Para.layers_dict[cur_dict["name"]] = cur_dict

    def _parse(self):
        dic = OrderedDict()
        dic["name"] = str(self.layer_para.name)
        dic["op"] = str(self.layer_para.op)
        dic["input"] = list(self.layer_para.input)

        dic["type"] = self.layer_para.attr["T"].type
        return dic


class Pad:

    def __init__(self, layer_para):
        super(Pad, self).__init__()
        self.layer_para = layer_para
        cur_dict = self._parse()
        Para.layers_dict[cur_dict["name"]] = cur_dict
        # print(layer_dict)

    def _parse(self):
        dic = OrderedDict()
        dic["name"] = self.layer_para.name
        dic["op"] = self.layer_para.op
        dic["input"] = self.layer_para.input
        dic["type"] = self.layer_para.attr["T"].type

        return dic


class Mul:

    def __init__(self, layer_para):
        super(Mul, self).__init__()
        self.layer_para = layer_para
        cur_dict = self._parse()
        Para.layers_dict[cur_dict["name"]] = cur_dict
        # print(layer_dict)

    def _parse(self):
        dic = OrderedDict()
        dic["name"] = self.layer_para.name
        dic["op"] = self.layer_para.op
        dic["input"] = self.layer_para.input
        dic["type"] = self.layer_para.attr["T"].type

        return dic


class Sub:

    def __init__(self, layer_para):
        super(Sub, self).__init__()
        self.layer_para = layer_para
        cur_dict = self._parse()
        Para.layers_dict[cur_dict["name"]] = cur_dict
        # print(layer_dict)

    def _parse(self):
        dic = OrderedDict()
        dic["name"] = self.layer_para.name
        dic["op"] = self.layer_para.op
        dic["input"] = self.layer_para.input
        dic["type"] = self.layer_para.attr["T"].type

        return dic


class Reshape:

    def __init__(self, layer_para):
        super(Reshape, self).__init__()
        self.layer_para = layer_para
        cur_dict = self._parse()

        Para.layers_dict[cur_dict["name"]] = cur_dict

    def _parse(self):
        dic = OrderedDict()
        dic["name"] = self.layer_para.name
        dic["op"] = self.layer_para.op
        dic["input"] = self.layer_para.input
        dic["type"] = self.layer_para.attr["T"].type
        dic["data_format"] = self.layer_para.attr["data_format"].s
        return dic
