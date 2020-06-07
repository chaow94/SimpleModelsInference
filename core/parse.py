import tensorflow as tf

from core.factory import support_ops
from core.parse_utils import *


class ParsePb:

    def __init__(self, model_filename):
        self.graph_def = tf.GraphDef()
        self.model_filename = model_filename

    def _parse(self):

        nets = OrderedDict()
        with open(self.model_filename, 'rb') as f:
            self.graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(self.graph_def, name='')

        with tf.Session(graph=graph) as sess:
            cnt = 0
            for n in sess.graph_def.node:
                if n.op in "Const":
                    eval(n.op + "(n)")

                if n.op in support_ops:
                    # op
                    if "/read" not in n.name or n.op == "Const":

                        eval(n.op + "(n)")
                        nets[cnt] = n.name
                        cnt = cnt + 1
                    else:
                        # parameter
                        eval(n.op + "(n)")

        return nets
