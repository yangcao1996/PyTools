from PIL import Image
import os.path as osp
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pkg_dir = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, osp.join(pkg_dir, 'pytools'))
import pytools
img = np.array(Image.open(osp.join(pkg_dir, 'data/3063.jpg')))
edge = np.array(Image.open(osp.join(pkg_dir, 'data/3063-edge.jpg')), dtype=np.float32)

overlayed = pytools.misc.overlay(img, edge, alpha=0.2)
plt.imshow(overlayed)
plt.savefig('overlayed.jpg')
