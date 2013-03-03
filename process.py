from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data, io, filter
from skimage import data
from skimage.filter import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import label, closing, square
from skimage.measure import regionprops

import utils
import IPython

utils.ipython_on_error()

pic = Image.open("test.jpg")
image = np.asarray(pic)[:,:,0]

thresh = threshold_otsu(image)
bw = closing(image > thresh, square(3))

label_image = label(bw)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(label_image, cmap='jet')

for region in regionprops(label_image, ['Area', 'BoundingBox']):

    # skip small images
    if region['Area'] < 100:
        continue

    # draw rectangle around segmented coins
    minr, minc, maxr, maxc = region['BoundingBox']
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)

plt.show()

IPython.embed()

