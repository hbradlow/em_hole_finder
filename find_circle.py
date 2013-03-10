from PIL import Image
import numpy as np
import math
import random

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

import scipy

import scipy.io
from skimage import data, io, filter
from skimage import data
from skimage.filter import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import label, closing, square, reconstruction, erosion
from skimage.measure import regionprops
import skimage.draw

from scipy.ndimage.morphology import grey_opening

import IPython

HOLE_RADIUS = 250

class Pipeline:
    def __init__(self,filename="data/test2.jpg",debug=True,downsample=20):
        self.image = Image.open(filename)
        self.data = np.asarray(self.image)[:,:]

        self.decimate(downsample)
        self.normalize()
        self.save()

        self.debug = debug
        self.point_inside_penalty = 50000000

    def open(self,window_size=(10,10)):
       self.data = np.max(self.data) - self.data
       self.data -= grey_opening(self.data,size=(5,5)) 
       self.data = np.max(self.data) - self.data

    def select_largest_component(self):
        max_component = max(self.component_sizes.items(),key=lambda x: x[1])[0]
        for (x,y),value in np.ndenumerate(self.data):
            if value!=max_component:
                self.data[x,y] = 0
            else:
                self.data[x,y] = 255

    def mask_original(self):
        for (x,y),value in np.ndenumerate(self.data):
            if value!=0:
                self.data[x,y] = self.saved_data[x,y]
            else:
                self.data[x,y] = 0

    def fill_holes(self):
        from skimage.morphology import square
        self.data = np.max(self.data)-self.data
        self.data = erosion(self.data,square(3))
        self.data = np.max(self.data)-self.data
        """
        mask = np.ones(self.data.shape)*np.min(self.data)
        seed = self.data

        self.data = reconstruction(seed, mask, method='erosion')
        """

    def connected_components_iterative(self):
        component_cache = np.zeros(self.data.shape)
        self.component_sizes = {}
        num_components = 0

        def get_neighbors(x,y,depth=1):
            for i in range(-depth,depth+1):
                for j in range(-depth,depth+1):
                    if i+x<self.data.shape[0] and j+y<self.data.shape[1] and i+x>=0 and j+y>=0 and (i != 0 or j != 0):
                        yield (i+x,j+y)

        self.num_components = 0

        pixel_stack = []
        for (x,y),value in np.ndenumerate(self.data):
            if not value:
                pixel_stack.append((x,y))
        while pixel_stack:
            x,y = pixel_stack.pop()
            if component_cache[x,y]!=0 or self.data[x,y]:
                continue

            found_component = 0

            for (nx,ny) in get_neighbors(x,y):
                if component_cache[nx,ny] != 0:
                    component_cache[x,y] = component_cache[nx,ny]
                    found_component = component_cache[nx,ny]
                    self.component_sizes[found_component] += 1
                    break

            if not found_component:
                self.num_components += 1
                found_component = self.num_components
                component_cache[x,y] = found_component
                self.component_sizes[found_component] = 1

            for (nx,ny) in get_neighbors(x,y):
                pixel_stack.append((nx,ny))

        self.data = component_cache
        print self.component_sizes


    def connected_components(self):
        component_cache = np.zeros(self.data.shape)
        self.component_sizes = {}
        num_components = 0

        def get_neighbors(x,y,depth=1):
            for i in range(-depth,depth+1):
                for j in range(-depth,depth+1):
                    if i+x<self.data.shape[0] and j+y<self.data.shape[1] and i+x>=0 and j+y>=0 and (i != 0 or j != 0):
                        yield (i+x,j+y)

        self.num_components = 0

        def process(x,y,depth):
            if component_cache[x,y]!=0 or self.data[x,y] or depth>900:
                return

            found_component = 0

            for (nx,ny) in get_neighbors(x,y):
                if component_cache[nx,ny] != 0:
                    component_cache[x,y] = component_cache[nx,ny]
                    found_component = component_cache[nx,ny]
                    self.component_sizes[found_component] += 1
                    break

            if not found_component:
                self.num_components += 1
                found_component = random.randrange(255)+1 #self.num_components
                component_cache[x,y] = found_component
                self.component_sizes[found_component] = 1

            for (nx,ny) in get_neighbors(x,y):
                process(nx,ny,depth+1)

        for (x,y),value in np.ndenumerate(self.data):
            process(x,y,0)

        self.data = component_cache
        print self.component_sizes

    def save(self):
        self.saved_data = self.data.copy()

    def show(self,circle=False):
        if circle:
            io.imshow(self.data_with_circle())
        else:
            io.imshow(self.data)
        io.show()

    def data_with_circle(self):
        image = self.data.copy()
        for (x,y) in zip(self.circle_x,self.circle_y):
            if x < image.shape[0] and y < image.shape[1] and x >= 0 and y >= 0:
                image[x,y] = 0
        return image

    def fit_circle(self,initialization=[-150,-150],maxiter=50):
        beta = scipy.optimize.fmin(self.cost(),initialization,maxiter=maxiter)
        self.circle_x,self.circle_y = skimage.draw.circle_perimeter(cx=int(beta[0]),cy=int(beta[1]),radius=int(HOLE_RADIUS))

    def blur(self,window_size=(10,10)):
        window = np.ones(window_size)
        window /= np.linalg.norm(window)
        self.data = scipy.signal.convolve2d(self.data, window)#, boundary='fill',fillvalue=np.max(self.data))

    def fit_plane(self):
        X,Y = np.mgrid[:self.data.shape[0],:self.data.shape[1]]
        x = X.flatten()
        y = Y.flatten()
        A = np.column_stack((np.ones(x.size), x, y))
        c, resid,rank,sigma = np.linalg.lstsq(A,self.data.flatten())
        self.c = c

    def plot_3d(self):
        fig = plt.figure()

        X,Y = np.mgrid[:self.data.shape[0],:self.data.shape[1]]

        normal = np.array([self.c[2],self.c[1],1])

        z = (-normal[0]*X - normal[1]*Y + self.c[0])/normal[2]

        print "HERE"
        ax = fig.add_subplot(1,1,1, projection='3d')
        surf = ax.plot_surface(X,Y,self.data)
        surf = ax.plot_surface(X,Y,z)
        print "HERE"

        a = np.average(z)
        for (x,y),value in np.ndenumerate(z):
            self.data[x,y] += a-value
        print "HERE"

        ax2 = fig.add_subplot(1,1,1, projection='3d')
        surf = ax2.plot_surface(X,Y,self.data)
        surf = ax2.plot_surface(X,Y,z)
        plt.show()



    def dilate(self):
        def get_neighbors(x,y,depth=1):
            for i in range(-depth,depth+1):
                for j in range(-depth,depth+1):
                    if i+x<self.data.shape[0] and j+y<self.data.shape[1] and i+x>=0 and j+y>=0 and (i != 0 or j != 0):
                        yield (i+x,j+y)

        copy_data = self.data.copy()

        for (x,y),value in np.ndenumerate(self.data):
            for (nx,ny) in get_neighbors(x,y):
                if not self.data[nx,ny]:
                    copy_data[x,y] = 0
                    break
        self.data = copy_data

    def threshold(self):
        """
            Simple threshold for intensity.
        """
        """
        for i,row in enumerate(self.data):
            for j,datum in enumerate(row):
                if datum < np.average(self.data):
                    self.data[i,j] = 0
                else:
                    self.data[i,j] = np.max(self.data)
        """
        self.data = filter.threshold_adaptive(self.data,100)

    def connectivity_threshold(self,threshold=24,depth=2):
        """
            Threshold based off of connectivity.
        """
        tmp_image = np.zeros(self.data.shape)
        for i,row in enumerate(self.data):
            for j,datum in enumerate(row):
                num_neighbors = 0
                for x in range(-depth,depth+1):
                    for y in range(-depth,depth+1):
                        if i+x<self.data.shape[0] and j+y<self.data.shape[1] and i+x>0 and j+y>0 and (x != 0 or y != 0):
                            if self.data[i+x,j+y] == 0:
                                num_neighbors += 1
                if num_neighbors>=threshold:
                    tmp_image[i,j] = 0
                else:
                    tmp_image[i,j] = 255
        self.data = tmp_image

    def crop(self,factor=8):
        self.data = self.data[factor:-factor,factor:-factor]

    def decimate(self,downsample=20):
        self.data = scipy.signal.decimate(self.data,downsample,axis=0,ftype='fir')
        self.data = scipy.signal.decimate(self.data,downsample,axis=1,ftype='fir')

    def movingaverage(self, window_size=5):
        window = np.ones(window_size)
        window /= np.linalg.norm(window)
        self.data = scipy.signal.convolve2d(self.data, window)

    def normalize(self):
        image_min = np.min(self.data)
        image_max = np.max(self.data)
        self.data = self.data - np.ones(self.data.shape)*image_min
        self.data *= (image_max)/(image_max-image_min)

    def save_to_file(self):
        rescaled = (255.0 / self.data.max() * (self.data - self.data.min())).astype(np.uint8)
        im = Image.fromarray(rescaled)
        im.save("output.jpg")
    def cost(self):
        def c(beta):
            def in_circle(x,y):
                return (x-beta[0])**2 + (y-beta[1])**2 < HOLE_RADIUS**2

            error = 0
            for x,row in enumerate(self.data):
                for y,point in enumerate(row):
                    if not point:
                        error += abs((x-beta[0])**2 + (y-beta[1])**2 - HOLE_RADIUS**2)
                        if in_circle(x,y):
                            error += self.point_inside_penalty

            if self.debug:
                print "Calculate cost", error, beta

            return error
        return c

p = Pipeline(downsample=20)

p.open()
p.blur()
#p.show()

#p.fit_plane()
#p.plot_3d()

p.crop(factor=15)
p.threshold()

#p.show()
p.fill_holes()
#p.show()

p.connected_components_iterative()
#p.show()

p.select_largest_component()
#p.show()

p.connected_components_iterative()
p.select_largest_component()

for i in range(10):
    p.dilate()
p.mask_original()
p.show()
p.save_to_file()
