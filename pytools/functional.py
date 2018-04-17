# =================================================
# * Licensed under The MIT License
# * Written by KAI-ZHAO (http://kaiz.xyz)
# =================================================
import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def laplace(x, mu=0, b=1):
  assert b != 0
  return 1 / (2 * b) * np.exp(-np.abs(x - mu) / b)

def norm01(x):
  assert isinstance(x, np.ndarray)
  xmax, xmin  = x.max(), x.min()
  if xmin < xmax:
    x -= xmin
    x /= xmax - xmin
  return x
