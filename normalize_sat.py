import cv2
import os
from skimage import io
import numpy as np

def normalize_saturation(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    mean_v = np.mean(v*1.0)
    v = (v / mean_v * 0.6529);
    #v = np.clip(v, 0, 255).astype(np.uint8)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)