import sys
sys.path.append("../")
from sample.model import MyModel
from Visualizer import Visualizer
import pickle
import numpy as np
import torch
import cv2


# img_np = cv2.imread("pooh1.jpg")
# img_np = img_np.reshape(1, img_np.shape[0], img_np.shape[1], img_np.shape[2]) / 255
img_np = np.random.randint(0, 256, (1, 100, 100, 3)) / 255
img = torch.tensor(img_np)

model = MyModel()
visualizer = Visualizer(model)
visualizer.describe_layers()
visualizer.describe_kernel(layer_name=input("Please input layer name for describe >>> "), row=3, column=3)
outputs = visualizer.generate_output(input("Please input layer name >>> "), img, is_show=True)
outputs = outputs.view(1, outputs.size[0], outputs.size[1], outputs.size[2])
outputs = visualizer.generate_output(input("Please input layer name >>> "), outputs, is_show=True)
