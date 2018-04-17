import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
FIG_DIR = os.path.join(THIS_DIR, 'figures')
if not os.path.isdir(FIG_DIR):
  os.makedirs(FIG_DIR)
y = 1
x = np.arange(-0.1, 1.1, 0.001)
tp = y*x
fp = y*(1-x)
fn = (1-y) * x
p = tp / (tp+fp)
r = tp / (tp + fn)
f = 2*p*r/(p+r)
plt.plot(1-f, x)
plt.hold(True)
plt.plot(np.arange(0.001, 1, 0.001), -np.log(np.arange(0.001, 1, 0.001)))
plt.grid(True)
plt.title("Loss curve ($y=1$)")
plt.legend(["Fmeasure-loss", "CELoss"])
plt.xlabel("$\\hat{y}$")
plt.ylabel("$\\mathcal{L}(\\hat{y}, y)$")
plt.savefig(os.path.join(FIG_DIR, 'loss-curve.pdf'))
plt.savefig(os.path.join(FIG_DIR, 'loss-curve.svg'))
plt.savefig(os.path.join(FIG_DIR, 'loss-curve.png'))
