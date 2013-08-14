import sys
from optparse import OptionParser

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

def find_mask(filename,show=False,outfile="output.jpg", downsample=20, compsizethresh=50,adapthresh=500,blur=10,dilation=10,erosion=1):
    p = Pipeline(filename=filename, downsample=20,dilation=10,erosion=1)

    p.open() #perform background subtraction
    blurnum = int(blur)
    blurary = (blurnum,blurnum)
    p.blur(blurary) #blur the image
    p.crop(factor=8) #crop out the black borders that are produced by the blur
    adapthreshnum = int(adapthresh)
    p.threshold(adapthreshnum) #do adaptive thresholding
    p.erode() #erode the thresholded image

    p.connected_components_iterative() #calculate the connected components
    """
    try:
        p.check_circularity() #check to make sure everything makes sense so far
    except:
        print "I think this one doesn't have the hole in it..."
        #p.save_to_file(p.saved_data,filename=outfile) #saved the masked image to a file
        return np.ones(p.data.shape) #return an empty mask
    """
    p.select_largest_component() #select the largest connected component

    p.connected_components_iterative(full=False) #calculate the connected components of the inverted image
    #p.select_largest_component() #selected the largest connected components of the inverted image (this should now be the interior of the hole)
    p.select_largest_component()

    p.dilate() #dilate the mask
    p.mask_original() #mask the original image with the calculated mask
    p.save_to_file(out=p.data, filename=outfile) #saved the masked image to a file
    if show:
        p.show(data=np.concatenate((p.saved_data,p.data),axis=1)) #show the mask for debugging

    return p.data

def printHelp():
    print "hole_finder.py -i <input file>"

def main(argv):
    parser = OptionParser()
    parser.add_option('--ifile',dest='ifile',default="test1.jpg", help='The input micrograph file (jpg)')
    parser.add_option('--ofile',dest='ofile', help='The input micrograph file (jpg)')
    parser.add_option('--downsample',dest='downsample',default=20, help='Downsampling rate.')
    parser.add_option('--compsizethresh',dest='compsizethresh',default=50, help='Component size thresholding')
    parser.add_option('--adapthresh',dest='adapthresh',default=500, help='Adaptive thresholding factor')
    parser.add_option('--blur',dest='blur',default=10, help='Blur factor window size')
    parser.add_option('--dilation',dest='dilation',default=10, help='Dilation factor')
    parser.add_option('--erosion',dest='erosion',default=1, help='Erosion factor')

    (options, args) = parser.parse_args(argv)
    params = {}
    for i in parser.option_list:
        if isinstance(i.dest, str):
	    params[i.dest] = getattr(options, i.dest)
    
    if not params['ofile']:
        params['ofile'] = params['ifile'] + "_out.jpg"

    find_mask( params['ifile'], outfile=params['ofile'], downsample=params['downsample'], compsizethresh=params['compsizethresh'], adapthresh=params['adapthresh'], blur=params['blur'], dilation=params['dilation'], erosion=params['erosion'] )


if __name__=="__main__":
    main(sys.argv[1:])


