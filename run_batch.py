from find_mask import find_mask
import os

l = [n for n in os.walk("jpgs")]
for f in l[0][2]:
    print f
    if "jpg" in f:
        outfile = "output/" + f.split(".")[0] + "_out" + ".jpg"
        find_mask("jpgs/" + f,outfile=outfile,show=False)
