from PIL import Image
import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import scipy

import scipy.io
from skimage import data, io, filter
from skimage import data
from skimage.filter import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import label, closing, square
from skimage.measure import regionprops
import skimage.draw

import utils
import IPython

radius = 55000

#utils.ipython_on_error()

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]
def movingaverage(interval, window_size):
    window = np.ones(window_size)
    window /= np.linalg.norm(window)
    return scipy.signal.convolve2d(interval, window)

def rescale(image):
    image_min = np.min(image)
    image_max = np.max(image)
    image = image - np.ones(image.shape)*image_min
    image *= (image_max)/(image_max-image_min)
    return image

def cost(beta):
    def in_circle(x,y):
        return (x-beta[0])**2 + (y-beta[1])**2 < beta[2]
    if in_circle(0,0) and in_circle(0,height) and in_circle(width,0) and in_circle(width,height):
        return 0
    num_in = 0
    num_out = 0
    in_total = 0
    out_total = 0
    for i,row in enumerate(image):
        for j,datum in enumerate(row):
            if in_circle(i,j):
                num_in += 1
                in_total += datum
            else:
                num_out += 1
                out_total += datum
    if num_in ==0 or num_out ==0: 
        return 0
    average_in = in_total/float(num_in)
    average_out = out_total/float(num_out)
    print "AVERAGES: ",average_in,average_out
    print "Num in/out: ",num_in, num_out
    factor = .001
    c = abs(average_in - average_out) + factor*abs(num_in-num_out)
    print "COST: ",beta,c
    return c

def cost(beta):
    beta = np.append(beta,radius)
    def in_circle(x,y):
        return (x-beta[0])**2 + (y-beta[1])**2 < beta[2]
    error = 0
    factor = 1000000
    for i,row in enumerate(image):
        for j,point in enumerate(row):
            if not point:
                x = i
                y = j
                error += abs((x-beta[0])**2 + (y-beta[1])**2 - beta[2])
                if in_circle(x,y):
                    error += factor
    print "Calculate cost", error, beta
    return error

def fit_circle(data):
    xs = []
    ys = []
    for i,row in enumerate(data):
        for j,datum in enumerate(row):
            if datum == 0:
                xs.append(i)
                ys.append(j)
    xs = np.array(xs)
    ys = np.array(ys)
    scipy.io.savemat('xs.mat', mdict={'arr': xs})
    scipy.io.savemat('ys.mat', mdict={'arr': ys})
    a = np.array([xs,ys,np.ones(xs.shape)]) / (-xs**2 - ys**2)
    print "A: ",a
    x = .5* a[0]
    x = .5* a[1]
    r = math.sqrt((a[0]**2+a[1]**2)/4-a[2])
    return (x,y,r)

pic = Image.open("data/test2.jpg")
image = np.asarray(pic)[:,:]
saved_image = np.asarray(pic)[:,:]
image = scipy.signal.decimate(image,20,axis=0)
image = scipy.signal.decimate(image,20,axis=1)
image = rescale(image)
image = image[8:-8,8:-8]
saved_image = saved_image[8:-8,8:-8]

data = {}
for row in image:
    for datum in row:
        d = int(datum)
        if d in data:
            data[d] += 1
        else:
            data[d] = 1

#plt.bar(data.keys(),data.values())
#plt.show()
saved_image = image.copy()
for i,row in enumerate(image):
    for j,datum in enumerate(row):
        if datum < np.average(image):
            image[i,j] = 0
        else:
            image[i,j] = np.max(image)

new_image = np.zeros(image.shape)
#io.imshow(image)
#io.show()
for i,row in enumerate(image):
    for j,datum in enumerate(row):
        thresh = 24 
        r = 2
        for x in range(-r,r+1):
            for y in range(-r,r+1):
                if i+x<image.shape[0] and j+y<image.shape[1] and i+x>0 and j+y>0 and (x != 0 or y != 0):
                    try:
                        if image[i+x,j+y] == 0:
                            thresh -= 1
                    except IndexError:
                        print (i+x,j+y)
                        print image.shape
                        print (i+x,j+y)<image.shape
                        exit()
        if thresh<=0:
            new_image[i,j] = 0
        else:
            new_image[i,j] = 255
image = new_image
#io.imshow(new_image)
#io.show()

#image = movingaverage(image,(5,5))
height,width = image.shape
print "SHAPE: ",image.shape
#io.imshow(image)
#io.show()
#beta = fit_circle(image)
#print "CIRCLE: ",beta
beta = scipy.optimize.fmin(cost,[-150,-150],maxiter=200)
beta = np.append(beta,radius)

circle_x,circle_y = skimage.draw.circle_perimeter(cx=int(beta[0]),cy=int(beta[1]),radius=int(math.sqrt(beta[2])))

#saved_image = image
for (x,y) in zip(circle_x,circle_y):
    if x < saved_image.shape[0] and y < saved_image.shape[1] and x >= 0 and y >= 0:
        saved_image[x,y] = 0

io.imshow(saved_image)
io.show()
