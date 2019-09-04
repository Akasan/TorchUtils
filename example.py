from sample.model import MyModel
from visualizer import Visualizer
import pickle
import numpy as np
import torch
import cv2


img_np = np.random.randint(0, 256, (100, 100, 3)) / 255
img = torch.tensor(img_np)

model = MyModel()
visualizer = Visualizer(model)
visualizer.separate_layers()
visualizer.describe_layers()
output = visualizer.check_output(input("Please input layer name >>> "), img)
output = np.array(output).astype(np.uint8)

cv2.imshow("original_image", img_np * 255)
cv2.imshow("output_image", output * 255)
