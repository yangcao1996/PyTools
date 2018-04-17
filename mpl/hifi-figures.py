# =================================================
# * Written by KAI-ZHAO (http://kaiz.xyz)
# =================================================
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
print(THIS_DIR)
if not os.path.isdir(os.path.join(THIS_DIR, 'figures')):
  os.makedirs(os.path.join(THIS_DIR, 'figures'))

data1 = np.array([.387, .52, .594, .649, .678, .677, .679])
data2 = np.array([.443, .634, .694, .699, .701, .703, .703])
data3 = np.array([.507, .692, .723, .717, .723, .722, .724])
fig = plt.figure(figsize=(6,4))
ax = plt.gca()
xticks = np.arange(2, 9, 1)
plt.xticks(xticks)
plt.grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
plt.ylim(0.38, 0.75)
plt.xlim(2, 8.2)
xmin, xmax  = 2, 9
xs = np.arange(xmin, xmax)
plt.plot(xs, data1, linewidth=2, color='b')
plt.plot(xs, data2, linewidth=2, color='green')
plt.plot(xs, data3, linewidth=2, color='r')
plt.legend(['Direct (Fig.2 (c))', 'Hi-Fi-1', 'Hi-Fi-2'], fontsize=16)
plt.plot(xs, data1, '*', color='black')
plt.plot(xs, data2, '*', color='black')
plt.plot(xs, data3, '*', color='black')
plt.xlabel('Iterations (X5000)', fontsize=16)
plt.ylabel('F-measure', fontsize=16)
plt.title('Convergence Analysis', fontsize=16)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.set_aspect(10)

plt.savefig('figures/fig-convergence.pdf')
plt.savefig('figures/fig-convergence.svg')
plt.savefig('figures/fig-convergence.jpg')

fig.clear()
minor_ysticks = np.arange(0.67, 0.710, 0.005)
major_ysticks = np.arange(0.67, 0.710, 0.1)
xmin, xmax  = 2, 5
ymin, ymax = 0.675, 0.705
xs = np.arange(xmin, xmax+1)
data1 = np.array([.703, .689, .690, .679 ])
plt.xlim(xmin-0.2, xmax+0.2)
plt.ylim(ymin, ymax)
plt.xticks(np.arange(xmin, xmax+1, 0.5))
plt.yticks(minor_ysticks)
plt.plot([2,3,4,5], data1, linewidth=2, color='red')
plt.plot([2,3,4,5], data1, '*', color='black')
plt.grid(alpha=1, linestyle='dotted', linewidth=2, color='black')
plt.xlabel("$K$", fontsize=16)
plt.ylabel("F-measure", fontsize=16)
plt.savefig(os.path.join(THIS_DIR, 'figures', 'fig-mechanism.eps'))
plt.savefig(os.path.join(THIS_DIR, 'figures', 'fig-mechanism.pdf'))
plt.savefig(os.path.join(THIS_DIR, 'figures', 'fig-mechanism.jpg'))
plt.savefig(os.path.join(THIS_DIR, 'figures', 'fig-mechanism.svg'))
