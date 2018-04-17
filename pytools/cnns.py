# =================================================
# * Licensed under The MIT License
# * Written by KAI-ZHAO (http://kaiz.xyz)
# =================================================
import numpy as np
from PIL import Image
from os.path import isfile
from skimage import color as cl, img_as_float
import matplotlib.pyplot as plt

def caffe_blob2image(blob, mean=[104.00699, 116.66877, 122.67892]):
  assert isinstance(blob, np.ndarray)
  assert blob.ndim == 4
  mean = np.array(mean)
  images = blob.transpose((0, 2, 3, 1))
  images += mean
  images = np.squeeze(images).astype(np.uint8)
  return images
