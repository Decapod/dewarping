#!/usr/bin/python
import os
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import sys
import shutil
import matplotlib
matplotlib.use('WxAgg')
from pylab import *
from scipy.ndimage.filters import gaussian_filter

print "This tool is used to set background and string color."
print "It will show the picture and then expects two mouse clicks."
print "The first click should be centered on the string, while the second click should be centered in the background."
print "Determined colors are written to disk and used by dewarping."

class ColSel:
    def __init__(self,image,sigma=15):
        self.image = image
        self.smoothed = gaussian_filter(image,(sigma,sigma,0))
        self.sepcol = None
        self.bgcol = None
    def __save__(self,path=".colors"):
        f = open(".colors",'wb')
        f.write("%i %i %i\n"%(self.sepcol[0],self.sepcol[1],self.sepcol[2]))
        f.write("%i %i %i\n"%(self.bgcol[0],self.bgcol[1],self.bgcol[2]))
        f.close()
    def __call__(self,event):
        if event.inaxes:
            cx,cy = event.xdata,event.ydata
            if self.sepcol==None:
                pix = self.image[cy,cx]
                if self.image.dtype=='float32':
                    pix*=255
                print "Page separator color:",pix
                self.sepcol = pix
            elif self.bgcol==None:
                pix = self.smoothed[cy,cx]
                if self.image.dtype=='float32':
                    pix*=255
                print "Background color:",pix
                self.bgcol = pix
                self.__save__()
                print "Persisted colors to disk."
                shutil.rmtree(os.environ['MPLCONFIGDIR'])
                sys.exit(0)

if len(sys.argv)!=2:
    print "Usage:",sys.argv[0],"<image>"
    sys.exit(1)

ioff()
image = imread(sys.argv[1])
imshow(image)
cs = ColSel(image)
connect('button_press_event',cs)
show()

