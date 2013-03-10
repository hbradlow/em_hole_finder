#PIL
from PIL import Image

#numpy
import numpy as np

#matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#scipy
import scipy
import scipy.signal
from scipy.ndimage.morphology import grey_opening

#skimage
from skimage import data, io, filter
from skimage.morphology import label, closing, square, reconstruction, erosion, dilation, square

#ipython
import IPython

class Pipeline:
    def __init__(self,filename="data/test2.jpg",debug=False,downsample=20):
        self.image = Image.open(filename)
        self.data = np.asarray(self.image)[:,:]

        self.decimate(downsample)
        self.normalize()
        self.save()

        self.debug = debug

    def open(self,window_size=(10,10)):
        """
            Perform the opening of this image.

            http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.morphology.grey_opening.html
        """
        self.data = np.max(self.data) - self.data
        self.data -= grey_opening(self.data,size=(5,5)) 
        self.data = np.max(self.data) - self.data

    def select_largest_component(self):
        """
            Color the largest connected component white, and everything else black.
        """
        max_component = max(self.component_sizes.items(),key=lambda x: x[1])[0]
        for (x,y),value in np.ndenumerate(self.data):
            if value!=max_component:
                self.data[x,y] = 0
            else:
                self.data[x,y] = 255

    def mask_original(self):
        """
            Mask the original image with the computed mask in self.data.
        """
        for (x,y),value in np.ndenumerate(self.data):
            if value!=0:
                self.data[x,y] = self.saved_data[x,y]
            else:
                self.data[x,y] = 0

    def erode(self):
        """
            Erode the black regions of the image.

            NOTE:
            The image is first inverted because the erosion function erodes white regions.
        """
        self.data = np.max(self.data)-self.data
        self.data = erosion(self.data,square(3))
        self.data = np.max(self.data)-self.data

    def dilate(self):
        """
            Dilate the black regions of the image.

            NOTE:
            The image is first inverted because the dilation function dilates white regions.
        """
        self.data = (1.0 / self.data.max() * (self.data - self.data.min())).astype(np.uint8) #convert the values to floats between 0 and 1

        self.data = np.max(self.data)-self.data
        self.data = dilation(self.data,square(20))
        self.data = np.max(self.data)-self.data

    def connected_components_iterative(self):
        """
            Iteratively compute the connected components of the iamge.

            Output:
            self.data will contain numbers from 0-num_components.
            0 means that a pixel is not in any component.
            Any other number refers to which component the pixel is part of.
        """
        component_cache = np.zeros(self.data.shape)
        self.component_sizes = {}
        num_components = 0

        def get_neighbors(x,y,depth=1):
            """
                Compute the neighbors of this pixel.
            """
            for i in range(-depth,depth+1):
                for j in range(-depth,depth+1):
                    if i+x<self.data.shape[0] and j+y<self.data.shape[1] and i+x>=0 and j+y>=0 and (i != 0 or j != 0):
                        yield (i+x,j+y)

        self.num_components = 0

        pixel_stack = [] #a stack of pixels to process
        for (x,y),value in np.ndenumerate(self.data): #first fill the stack with all the pixels in the image
            if not value:
                pixel_stack.append((x,y))

        while pixel_stack: #process all the pixel
            x,y = pixel_stack.pop()
            if component_cache[x,y]!=0 or self.data[x,y]: #dont process this pixel if it is has already been processed, or it is not black.
                continue

            found_component = 0

            for (nx,ny) in get_neighbors(x,y): 
                if component_cache[nx,ny] != 0: #if any of this pixel's neighbors are in a component, then this pixel is also in that component
                    component_cache[x,y] = component_cache[nx,ny]
                    found_component = component_cache[nx,ny]
                    self.component_sizes[found_component] += 1
                    break

            if not found_component: #if none of this pixel's neighbors are in a component, then make a new component
                self.num_components += 1
                found_component = self.num_components
                component_cache[x,y] = found_component
                self.component_sizes[found_component] = 1

            for (nx,ny) in get_neighbors(x,y): #process all of my neighbors
                pixel_stack.append((nx,ny))

        self.data = component_cache
        if self.debug:
            print self.component_sizes

    def save(self):
        """
            Save a copy of the original data.
        """
        self.saved_data = self.data.copy()

    def show(self,circle=False):
        """
            Show the data in a debugging window.
        """
        if circle:
            io.imshow(self.data_with_circle())
        else:
            io.imshow(self.data)
        io.show()

    def blur(self,window_size=(10,10)):
        """
            Blur the image.
        """
        window = np.ones(window_size)
        window /= np.linalg.norm(window)
        self.data = scipy.signal.convolve2d(self.data, window)#, boundary='fill',fillvalue=np.max(self.data))

    def fit_plane(self):
        """
            Find the best fit plane to the data and save the parameter in self.c.

            self.c = [a,b,c]
            a + by + cx = z
        """
        X,Y = np.mgrid[:self.data.shape[0],:self.data.shape[1]]
        x = X.flatten()
        y = Y.flatten()
        A = np.column_stack((np.ones(x.size), x, y))
        c, resid,rank,sigma = np.linalg.lstsq(A,self.data.flatten())
        self.c = c

    def plot_3d(self):
        """
            Plot the data points in 3d.
        """
        fig = plt.figure()
        X,Y = np.mgrid[:self.data.shape[0],:self.data.shape[1]]
        ax = fig.add_subplot(1,1,1, projection='3d')
        surf = ax.plot_surface(X,Y,self.data)
        plt.show()

    def threshold(self):
        """
            Simple threshold for intensity.
        """
        self.data = filter.threshold_adaptive(self.data,100)

    def crop(self,factor=8):
        """
            Crop the borders of the image.
        """
        self.data = self.data[factor:-factor,factor:-factor]

    def decimate(self,downsample=20):
        """
            Decimate the image by a downsampling factor.
        """
        self.data = scipy.signal.decimate(self.data,downsample,axis=0,ftype='fir')
        self.data = scipy.signal.decimate(self.data,downsample,axis=1,ftype='fir')

    def normalize(self):
        """
            Rescale the values of the image.
        """
        image_min = np.min(self.data)
        image_max = np.max(self.data)
        self.data = self.data - np.ones(self.data.shape)*image_min
        self.data *= (image_max)/(image_max-image_min)

    def save_to_file(self):
        """
            Save the data to an image file.
        """ 
        rescaled = (255.0 / self.data.max() * (self.data - self.data.min())).astype(np.uint8)
        im = Image.fromarray(rescaled)
        im.save("output.jpg")


if __name__=="__main__":
    import sys
    if len(sys.argv)>1:
        filename = sys.argv[1]
    else:
        filename = "data/test2.jpg"
    p = Pipeline(downsample=20,filename=filename)

    p.open()
    p.blur()

    p.crop(factor=15)
    p.threshold()

    p.erode()

    p.connected_components_iterative()
    p.select_largest_component()

    p.connected_components_iterative()
    p.select_largest_component()

    p.dilate()
    p.mask_original()
    p.show()
    p.save_to_file()
