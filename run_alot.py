import time
from find_mask import find_mask
import cProfile
cProfile.run("find_mask('data/test2.jpg',show=False)")
exit()
for i in range(10):
    start_time = time.time()
    find_mask("data/test2.jpg",show=False)
    print "Took",time.time()-start_time,"seconds to complete"
