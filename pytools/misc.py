import numpy as np
from skimage import color, io, img_as_float

def overlay(img, mask, alpha=0.6):
  mask = norm01(mask)
  img = img_as_float(img)
  print(img.max(), img.min())
  if mask.ndim == 2:
    mask = np.dstack(mask, mask, mask)
  if img.ndim == 2:
    img = np.dstack(img, img, img)
  img_hsv = color.rgb2hsv(img)
  mask_hsv = color.rgb2hsv(mask)
  img_hsv[..., 0] = mask_hsv[..., 0]
  img_hsv[..., 1] = mask_hsv[..., 1] * alpha
  return color.rgb2hsv(img_hsv)

def overlay1(img, mask, color=[255, 0, 0], solid=True):
  """
  overlay binary/heatmap to image for visualization
  """
  assert img.ndim == 2 or img.ndim == 3
  assert mask.ndim == 2
  if img.ndim == 2:
    img = img[:, :, np.newaxis]
  img, mask = img.astype(np.float), mask.astype(np.float)
  output = img.copy()
  color = np.array(color, dtype=np.float32)
  for x in range(img.shape[1]):
    for y in range(img.shape[0]):
      if mask[y, x] !=0:
        if solid:
          output[y, x, :] = color
        else:
          output[y, x, :] = (color * mask[y, x]) # TODO
  return output.astype(np.uint8)
  
def norm01(x):
  """
  Normalize heatmap into [0, 1]
  """
  assert type(x) is np.ndarray
  x = x.astype(np.float)
  return (x-x.min()) / x.max()

def norm255(x):
  """
  Normalize heatmap into [0, 255]
  """
  assert type(x) is np.ndarray
  return norm01(x) * 255

def blob2im(blob):
  assert isinstance(blob, np.ndarray)
  im = np.squeeze(blob[0, :, :, :])
  im = im.transpose((1, 2, 0))
  im += np.array((104.00699, 116.66877, 122.67892))
  im = im.astype(np.uint8)
  return im
