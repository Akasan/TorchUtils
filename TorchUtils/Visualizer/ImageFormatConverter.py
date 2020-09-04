import cv2
import numpy as np



def pil_to_opencv(img):
    img = img.transpose((1, 2, 0))
    img = img[:, :, ::-1]
    return img
