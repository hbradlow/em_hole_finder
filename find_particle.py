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

from pipeline import Pipeline

def find_particle(filename,show=True):
    from find_mask import find_mask
    hole_mask = find_mask(filename,show=False)

    p = Pipeline(downsample=20,filename=filename)

    #p.show()

    p.open()
    p.crop(factor=15)
    p.threshold() #do adaptive thresholding

    #p.show()

    p.connected_components_iterative(full=False) #calculate the connected components
    p.threshold_component_size()
    #p.show() #show the mask for debugging
    p.data = np.max(p.data) - p.data
    p.erode(factor=1)
    p.erode(factor=1)
    p.show()

    #p.show() #show the mask for debugging

    p.subtract(hole_mask)

    p.show()

    p.convex_hull_per_component()

    p.mask_original()
    p.show() #show the mask for debugging

if __name__=="__main__":
    import sys
    if len(sys.argv)>1:
        filename = sys.argv[1]
    else:
        filename = "data/dox.jpg"
    find_particle(filename)
