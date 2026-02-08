import cv2
import numpy as np

class ROSImgUtils():
    def __init__(self):
        pass

    def decodeImage(
        self,
        input: np.ndarray):

        cv2.imwrite('sample.jpg', input)