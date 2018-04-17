import os, sys
from os.path import join, abspath, dirname, isdir, split, splitext
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
pytools_dir = abspath(join(dirname(__file__), '..'))
sys.path.insert(0, pytools_dir)
from pytools.image import overlay, imread, imwrite, isimg, norm01
from pytools.matlab import bwdist
import numpy as np
overlayed_dir = join(pytools_dir, 'data/overlayed')
if not isdir(overlayed_dir):
  os.makedirs(overlayed_dir)
imgs = os.listdir(join(pytools_dir, 'data/images/'))
print("Performing image overlaying ...")
for i in imgs:
  fn, _ = splitext(i)
  im = imread(join(pytools_dir, 'data/images/', fn+'.jpg'))
  h, w, _ = im.shape
  mask = np.zeros((h, w), dtype=np.float32)
  y1, x1, r = h//2, w//2, 20
  mask[y1-r:y1+r, x1-r:x1+r] = 1
  mask = bwdist(mask)[0]
  mask = np.exp(-0.1 * mask)
  overlayed_im = overlay(im, mask)
  fig, axes = plt.subplots(1,3)
  axes[0].imshow(im)
  axes[0].set_title("Image")
  axes[1].imshow(mask, cmap=cm.Greys_r)
  axes[1].set_title("Mask")
  axes[2].imshow(overlayed_im, cmap=cm.Greys_r)
  axes[2].set_title("Overlayer")
  plt.savefig(join(overlayed_dir, fn+'.jpg'))
print("Done! Results are saved in '%s'." % (overlayed_dir))
