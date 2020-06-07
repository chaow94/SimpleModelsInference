import cv2
import numpy as np
import time

from core.nets import Net

cat = cv2.imread('cat.jpg')
cat = cv2.resize(cat, (224, 224))

cat = np.expand_dims(cat, 0)
print(cat.shape)

t1 = time.time()
model_filename = "./models/resnet50_fp32_pretrained_model.pb"

nets = Net(cat, model_filename,(1, 224, 224, 3), 'input', "test")

res = nets.forward()
t2 = time.time()
print("infer cost time {} s.".format(t2-t1))
print(res)
print(np.argmax(res))
