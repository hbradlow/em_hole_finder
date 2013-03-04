from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import scipy

from skimage import data, io, filter
from skimage import data
from skimage.filter import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import label, closing, square
from skimage.measure import regionprops

import utils
import IPython

#utils.ipython_on_error()

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]
def movingaverage(interval, window_size):
    window = np.ones(window_size)
    window /= np.linalg.norm(window)
    return scipy.signal.convolve2d(interval, window)

pic = Image.open("data/test2.jpg")
image = np.asarray(pic)[:,:,0]
image = movingaverage(image,(10,10))
image = movingaverage(image,(10,10))
image = movingaverage(image,(10,10))
image = movingaverage(image,(10,10))
io.imshow(image)
io.show()
image = filter.threshold_adaptive(image,200)
#image = movingaverage(image,(10,10))
#image = filter.canny(image,10)
#image = filter.tv_denoise(image)

io.imshow(image)
io.show()

#IPython.embed()
