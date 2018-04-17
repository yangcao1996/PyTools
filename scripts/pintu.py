from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from os.path import join
from pytools.image import isimg
# options
#======================================================================
ngroups = 4
H = 200 # height of every component image (pixel)
W = 3200 # width of synthesised image (pixel)
marginv = 40 # vertical margin (pixel)
# margin of generated picture
margin_top, margin_right, margin_bottom, margin_left = 20, 0, 10, 150
all_imgs = []
im_dir = '/media/data2/sk-results/pintu'
subpaths = ["GT_PASCAL", "HiFi_PASCAL", "SRN_PASCAL", "FSDS_PASCAL"]
#======================================================================
assert len(subpaths) == ngroups
imgs = [i for i in os.listdir(join(im_dir, subpaths[0])) if isimg(i)]
imgs.sort()
nimgs = len(imgs)
widths = np.zeros((nimgs,), dtype=np.int)
# prepare images
print("Preparing images...")
for i in range(ngroups):
  all_imgs.append([])
  for idx, j in enumerate(imgs):
    im = Image.open(join(im_dir, subpaths[i], j))
    w, h = im.size
    r = np.float(H) / h
    w1 = np.int(w * r)
    widths[idx] = w1
    im = np.array(im.resize((w1, H)), dtype=np.uint8)
    all_imgs[i].append(im)
print("Prepared %d images into %d groups." % (ngroups * nimgs, ngroups))

rows = [[]]
w1 = np.int(0)
row_id = 0
for idx, w in enumerate(widths):
  if w1 + w >= W:
    # next row
    rows.append([])
    w1 = w
    row_id = row_id + 1
    rows[row_id].append(dict({"id": idx, "width": w}))
  else:
    w1 += w
    rows[row_id].append(dict({"id": idx, "width": w}))
# draw the picture
picture = np.ones((len(rows) * (H * ngroups + marginv), W, 3), dtype=np.uint8) * 255
ystart = 0
for ridx, r in enumerate(rows):
  # row
  for g in range(ngroups):
    # group
    yend = ystart + H
    xstart = 0
    # calculate horizontal margin
    w2 = 0
    for c in r:
      w2 = w2 + c['width']
    marginh = (W - w2) // (len(r) - 1)
    for cidx, c in enumerate(r):
      # column
      w = c['width']
      idx = c['id']
      xend = xstart + w
      picture[ystart:yend, xstart:xend, :] = all_imgs[g][idx]
      xstart = xstart + w + marginh
    ystart = ystart + H
  ystart = ystart + marginv
# add margin
picture = np.pad(picture, ((margin_top, margin_bottom), (margin_left, margin_right), (0, 0)), mode='constant',
constant_values=255)
picture = Image.fromarray(picture)
# draw text on image
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
draw = ImageDraw.Draw(picture)
y = margin_top + H // 2
x = 16
for r in range(len(rows)):
  draw.text((x, y), 'GT', (0,0,0), font=font)
  y = y + H - 1
  draw.text((x, y), 'HiFi', (0,0,0), font=font)
  y = y + H 
  draw.text((x, y), 'SRN', (0,0,0), font=font)
  y = y + H  
  draw.text((x, y), 'FSDS', (0,0,0), font=font)
  y = y + H  
  draw = ImageDraw.Draw(picture)
  y = y + marginv
picture.save('/tmp/square-objects-in-SYM-PASCAL.pdf')
picture.save('/tmp/square-objects-in-SYM-PASCAL.jpg')
# http://data.kaiz.xyz/hifi/sqr-objs-SYM-PASCAL.pdf
