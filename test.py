from find_mask import Pipeline

import time
start_time = time.time()

filename = "test_images/test1.jpg"
p = Pipeline(downsample=20,filename=filename)

p.open() #perform background subtraction
p.blur() #blur the image
p.crop(factor=15) #crop out the black borders that are produced by the blur
p.threshold() #do adaptive thresholding
p.erode() #erode the thresholded image

p.connected_components_iterative() #calculate the connected components
p.select_largest_component() #select the largest connected component

p.connected_components_iterative() #calculate the connected components of the inverted image
p.select_largest_component() #selected the largest connected components of the inverted image (this should now be the interior of the hole)

p.dilate() #dilate the mask
p.mask_original() #mask the original image with the calculated mask

duration = time.time()-start_time
print "Took",round(duration),"seconds to complete"

p.show() #show the mask for debugging
p.save_to_file() #saved the masked image to a file
