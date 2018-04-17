# Test pynms
import os, sys
from os.path import join, abspath, dirname, isdir, split
pytools_dir = abspath(join(dirname(__file__), '..'))
sys.path.insert(0, pytools_dir)
if not isdir(join(pytools_dir, 'data/edges-nms')):
  os.makedirs(join(pytools_dir, 'data/edges-nms'))
from scipy import io
from PIL import Image
import numpy as np
from pytools import nms
from pytools.image import imread, imwrite
imgs = [join(pytools_dir, 'data/edges', i) \
for i in os.listdir(join(pytools_dir, 'data/edges/')) if ".png" in
i]
for im in imgs:
  fn = split(im)[-1]
  # E = np.array(Image.open(im))
  E = imread(im)
  E1 = nms.pynms(E)
  # E1 = Image.fromarray((E1 * 255).astype(np.uint8))
  imwrite(E1 * 255, join(pytools_dir, 'data/edges-nms', fn))
  # E1.save(join(pytools_dir, 'data/edges-nms', fn))
