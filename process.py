from PIL import Image
import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import scipy

from skimage import data, io, filter
from skimage import data
from skimage.filter import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import label, closing, square
from skimage.measure import regionprops
import skimage.draw

import utils
import IPython

#utils.ipython_on_error()

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]
def movingaverage(interval, window_size):
    window = np.ones(window_size)
    window /= np.linalg.norm(window)
    return scipy.signal.convolve2d(interval, window)

def cost(beta):
    error = 0
    for i,row in enumerate(image[20:190,:]):
        for j,point in enumerate(row[20:190]):
            if point:
                x = i+20
                y = j+20
                error += ((x-beta[0])**2 + (y-beta[1])**2 - beta[2])**2
    print "Calculate cost", error, beta
    return error

pic = Image.open("data/test2.jpg")
image = np.asarray(pic)[:,:]
original_image = np.asarray(pic)[:,:]
image = scipy.signal.decimate(image,20,axis=0)
image = scipy.signal.decimate(image,20,axis=1)
print image.shape
image = movingaverage(image,(10,10))

image = filter.threshold_adaptive(image,100)
print image.shape
io.imshow(image)
io.show()
exit()

beta = scipy.optimize.fmin(cost,[100,100,5000],maxiter=10)
circle_x,circle_y = skimage.draw.circle_perimeter(cx=int(beta[0]),cy=int(beta[1]),radius=int(math.sqrt(beta[2])))
print "Center: ",beta[0],beta[1]
print "Radius: ",math.sqrt(beta[2])

IPython.embed()

for (x,y) in zip(circle_x,circle_y):
    if x < image.shape[0] and y < image.shape[1] and x >= 0 and y >= 0:
        original_image[x,y] = 0

#image = filter.canny(image,10)
#IPython.embed()
io.imshow(original_image)
io.show()
exit()
io.imshow(image)
io.show()
#image = movingaverage(image,(10,10))
#image = filter.canny(image,10)
#image = filter.tv_denoise(image)

io.imshow(image)
io.show()

#IPython.embed()
