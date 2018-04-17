# =================================================
# * Licensed under The MIT License
# * Written by KAI-ZHAO (http://kaiz.xyz)
# =================================================
import numpy as np
from PIL import Image
from os.path import isfile, splitext
from skimage import color as cl, img_as_float
import matplotlib.pyplot as plt

def isimg(path):
  exts = ['.png', '.jpg', '.bmp', '.jpeg']
  _, ext = splitext(path)
  if ext in exts:
    return True
  else:
    return False

def imread(path):
  assert isfile(path), "%s doesn't exist!" % path
  return np.array(Image.open(path), dtype=np.uint8)

def imwrite(im, path):
  assert isinstance(im, np.ndarray)
  im = np.squeeze(im)
  if im.ndim == 2:
    im = norm01(im) * 225
  elif im.ndim == 3:
    if im.max() <=1 and im.min() != im.max():
      im = norm01(im)
      im *= 255
  else:
    raise IOError("unsupported image shape %s" % str(im.shape))
  im = Image.fromarray(im.astype(np.uint8))
  im.save(path)

def norm01(x):
  """
  Normalize heatmap into [0, 1]
  """
  assert isinstance(x, np.ndarray)
  x = x.astype(np.float32)
  xmax, xmin  = x.max(), x.min()
  if xmin < xmax:
    x -= xmin
    x /= xmax - xmin
  return x

def norm255(x):
  """
  Normalize heatmap into [0, 255]
  """
  assert type(x) is np.ndarray
  return norm01(x) * 255

def extend_dim(im):
  if im.ndim >= 3:
    return im
  assert im.ndim == 2
  im = im[:,:,np.newaxis]
  im = np.repeat(im, 3, 2)
  return im

def overlay(im, mask, cmap="coolwarm", alpha=0.6):
  """
  Overlay a mask (mask) on a given image (im)
  im: image to be overlayed
  mask: heat map mask
  cmap: colormap (see 'https://matplotlib.org/examples/color/colormaps_reference.html' for details).
  alpha: transparency
  """
  assert isinstance(im, np.ndarray)
  assert isinstance(mask, np.ndarray)
  assert mask.ndim == 2
  assert im.ndim == 2 or im.ndim == 3
  h, w = mask.shape
  if im.ndim == 2:
    im = extend_dim(im)
  cmap = plt.get_cmap("coolwarm")
  mask = cmap(mask)
  mask = np.delete(mask, 3, 2)
  # im = img_as_float(im)
  im_hsv = cl.rgb2hsv(im)
  mask_hsv = cl.rgb2hsv(mask)
  im_hsv[..., 0] = mask_hsv[..., 0]
  im_hsv[..., 1] = mask_hsv[..., 1] * alpha
  im_masked = cl.hsv2rgb(im_hsv)
  if im_masked.max() <= 1:
    im_masked *= 255
  return im_masked.astype(np.uint8)
