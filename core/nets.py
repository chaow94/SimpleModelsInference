import time
from collections import OrderedDict

import numpy as np

from core.factory import OpFactory
from core.parse import ParsePb
from core.parse_utils import Para


class Net:

    def __init__(self, input, model_path, input_shape, input_node, output_node):
        self.input = input
        self.model_path = model_path
        self.input_shape = input_shape
        self.input_node = input_node
        self.output_node = output_node
        self.layers_dict = Para.layers_dict
        self.feature_maps = OrderedDict()

    def forward(self):
        self.input = np.resize(self.input, self.input_shape)
        t = time.time()
        sorted_nets = self.topological_sort()
        # print("sorted_nets: ", sorted_nets)
        print("sorted_nets cost : ", (time.time() - t) * 1000)

        y = self.input
        self.feature_maps[self.input_node] = y

        for idx, name in sorted_nets.items():
            op_factor = OpFactory(self.layers_dict[name]["op"])
            _op = op_factor._create_op(name, self.layers_dict)
            # single input
            t1 = time.time()
            inputs = []
            for _input in self.layers_dict[name]["input"]:
                if _input == self.input_node:
                    inputs.append(self.feature_maps[_input])
                else:
                    if ("/read" not in _input) and (self.layers_dict[_input]["op"] not in "Const"):
                        print("-===: ", _input)
                        inputs.append(self.feature_maps[_input])

            y = _op.forward(*inputs)
            self.feature_maps[name] = y

            t2 = time.time()
            print("{} cost {} ms.".format(name, (t2 - t1) * 1000))
        return y

    def topological_sort(self):
        t = time.time()

        nets = ParsePb(self.model_path)._parse()

        print("nets parse cost: ", (time.time() - t) * 1000)
        t = time.time()

        node_num = len(nets)
        indegree = {i: 0 for i in range(node_num)}
        adj = {i: [] for i in range(node_num)}
        name2idx = {v: int(k) for k, v in nets.items()}
        for i in range(node_num):
            _inputs = self.layers_dict[nets[i]]["input"]
            for j in range(len(_inputs)):
                if _inputs[j] == self.input_node:
                    continue
                else:
                    if (self.layers_dict[_inputs[j]]["op"] == "Identity" and "/read" in _inputs[j]) or \
                            self.layers_dict[_inputs[j]]["op"] == "Const":
                        continue
                    else:
                        indegree[i] = indegree[i] + 1
                        adj[name2idx[_inputs[j]]].append(i)

        res = OrderedDict()
        # indegree == 0
        indegree_eq_0 = [i for i in range(node_num) if indegree[i] == 0]
        count = 0  # 计数，记录当前已经输出的顶点数

        while indegree_eq_0:
            v = indegree_eq_0.pop(0)  # 从队列中取出一个顶点
            res[v] = nets[v]
            count = count + 1

            # 将所有v指向的顶点的入度减1，并将入度减为0的顶点入栈
            for top in adj[v]:
                if indegree[top] > 0:
                    indegree[top] = indegree[top] - 1

                if indegree[top] == 0:
                    indegree_eq_0.append(top)  # 若入度为0，则入栈

        print("topological_sort cost: ", (time.time() - t) * 1000)

        if count < node_num:
            # return False  # 没有输出全部顶点，有向图中有回路
            return None
        else:
            return res  # 拓扑排序成功
