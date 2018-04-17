import numpy as np
import scipy

def bwdist(x):
  """
  Identical to Matlab 'bwdist()'
  """
  from scipy import ndimage
  assert np.count_nonzero(x) > 0, 'Input All zero!'
  x = np.squeeze(x)
  x = np.logical_not(x)
  assert x.ndim == 2, 'Require 2d matrix'
  dist, idx = ndimage.morphology.distance_transform_edt(x, return_indices=True)
  return dist, idx

def nnz(x):
  assert isinstance(x, np.ndarray)
  return np.count_nonzero(x)

