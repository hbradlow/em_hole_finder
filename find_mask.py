#PIL
from PIL import Image

#numpy
import numpy as np

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

def find_mask(filename,show=False,outfile="output.jpg"):
    p = Pipeline(downsample=20,filename=filename)

    p.open() #perform background subtraction
    p.blur() #blur the image
    p.crop(factor=15) #crop out the black borders that are produced by the blur
    p.threshold() #do adaptive thresholding
    p.erode() #erode the thresholded image

    p.connected_components_iterative() #calculate the connected components
    """
    try:
        p.check_circularity() #check to make sure everything makes sense so far
    except:
        print "I think this one doesn't have the hole in it..."
        p.save_to_file(p.saved_data,filename=outfile) #saved the masked image to a file
        return np.ones(p.data.shape) #return an empty mask
    """
    p.select_largest_component() #select the largest connected component

    p.connected_components_iterative(full=False) #calculate the connected components of the inverted image
    #p.select_largest_component() #selected the largest connected components of the inverted image (this should now be the interior of the hole)
    p.select_largest_component()

    p.dilate() #dilate the mask
    p.mask_original() #mask the original image with the calculated mask
    p.save_to_file(out=np.concatenate((p.saved_data,p.data),axis=1),filename=outfile) #saved the masked image to a file
    if show:
        p.show(data=np.concatenate((p.saved_data,p.data),axis=1)) #show the mask for debugging

    return p.data

if __name__=="__main__":
    import sys
    if len(sys.argv)>1:
        filename = sys.argv[1]
    else:
        filename = "data/test2.jpg"
    find_mask(filename)
