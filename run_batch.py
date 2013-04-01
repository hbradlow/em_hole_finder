from find_mask import find_mask
import time
import os


l = [n for n in os.walk("jpgs")]
num = 0
total = 0
for f in l[0][2]:
    if "jpg" in f:
        start_time = time.time()

        outfile = "output/" + f.split(".")[0] + "_out" + ".jpg"
        find_mask("jpgs/" + f,outfile=outfile,show=False)

        duration = time.time()-start_time
        total += duration
        num += 1
        print "Average time:", str(total/num),"seconds"
