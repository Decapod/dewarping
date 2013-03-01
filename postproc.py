#!/usr/bin/python

from chelper import *
import ctypes as C
import warnings
import os
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import shutil
import multiprocessing
from pylab import *
import sys
from scipy.ndimage.filters import uniform_filter,gaussian_filter
from scipy.ndimage.interpolation import rotate,zoom
from scipy.ndimage import measurements
from scipy.interpolate import interp2d
from scipy.optimize import fmin
from scipy.misc import imsave

if len(sys.argv)!=3:
    print "Usage:",sys.argv[0],"<img in> <img out>"
    print "Postprocesses 3d dewarped output using text-lines."
    sys.exit(1)

warnings.filterwarnings('ignore')
img = imread(sys.argv[1])
gimg = mean(img,axis=2)
mask = gimg!=0
cropped = gimg
bg = cropped==0
croppedimg = img
print "read input image"

def binarize(img,sigma=150.0,k=0.15):
    img = img.copy()
    if img.dtype==dtype('uint8'):
        img /= 255.0
    if len(img.shape)==3:
        img = mean(img,axis=2)
    s1 = uniform_filter(ones(img.shape),sigma)
    sx = uniform_filter(img,sigma)
    sxx = uniform_filter(img**2,sigma)
    avg_ = sx/s1
    stddev_ = maximum(sxx/s1-avg_**2,0.0)**0.5
    s0,s1 = avg_.shape
    s0 = int(s0)
    s1 = int(s1)
    avg = zeros(img.shape)
    zoom(avg_,1.0,output=avg[:s0,:s1],order=0,mode='nearest')
    stddev = zeros(img.shape)
    zoom(stddev_,1.0,output=stddev[:s0,:s1],order=0,mode='nearest')
    R = amax(stddev)
    thresh = avg*(1.0+k*(stddev/R-1.0))
    return array(255*(img>thresh),'uint8')

binarized = binarize(cropped,k=0.15)
binarized[bg] = 255
binarized = 255-binarized
print "binarized image"

# remove big binarization artifacts (mainly caused by page separator string)
labels,_ = measurements.label(binarized)
objs = measurements.find_objects(labels)
for i,o in enumerate(objs):
    if o[0].stop-o[0].start<0.005*binarized.shape[0]:
        binarized[labels==i+1] = 0
    if o[0].stop-o[0].start>0.03*binarized.shape[0]:
        binarized[labels==i+1] = 0
    if o[1].stop-o[1].start>0.075*binarized.shape[1]:
        binarized[labels==i+1] = 0

blurred = gaussian_filter(binarized,(1,25))
thresh = amin(blurred)+0.15*(amax(blurred)-amin(blurred))
threshed = where(blurred>thresh,1,0)
print "applied anisotropic gaussian"

def getleft(obj,threshed=threshed):
    for x in range(obj[1].start,obj[1].stop):
        for y in range(obj[0].start,obj[0].stop):
            if threshed[y,x]==1:
                return (x,y)

def getright(obj,threshed=threshed):
    for x in range(obj[1].stop-1,obj[1].start,-1):
        for y in range(obj[0].start,obj[0].stop):
            if threshed[y,x]==1:
                return (x,y)

print "detecting pageframe...",
cands,n = measurements.label(threshed)
objs = measurements.find_objects(cands)
lefts,rights = [],[]
for o in objs:
    l = getleft(o)
    r = getright(o)
    if l and r:
        lefts.append(l)
        rights.append(r)
lefts,rights = array(lefts),array(rights)

widths = [norm(lefts[i]-rights[i]) for i in range(len(lefts))]
#shortlines = len([l for l in widths if l<0.75*mean(widths)])
#if shortlines>0.6*len(widths):
#    print "unable to reliably detect text-lines... (too short) aborting postprocessing!"
#    imsave(sys.argv[2],croppedimg)
#    sys.exit(0)
lefts_new,rights_new = [],[]
for i in range(len(lefts)):
    if widths[i]>0.8*mean(widths):
        lefts_new.append(lefts[i])
        rights_new.append(rights[i])
lefts,rights = array(lefts_new),array(rights_new)

pagemid = threshed.shape[1]/2
lnew,rnew = [],[]
leftmost = min([lefts[i][0] for i in range(len(lefts))])
rightmost = max([rights[i][0] for i in range(len(rights))])
for i in range(len(lefts)):
    #if lefts[i][0]<pagemid-0.2*threshed.shape[1] and rights[i][0]>pagemid+0.2*threshed.shape[1]:
    if lefts[i][0]<leftmost+0.15*threshed.shape[1] and \
       rights[i][0]>rightmost-0.15*threshed.shape[1]:
        lnew.append(lefts[i])
        rnew.append(rights[i])
lefts,rights = array(lnew),array(rnew)

threshval=0.025*threshed.shape[1]
def linefit(points,thresh=threshval):
    cands = points.copy()
    cands_old = []
    while not all(cands==cands_old):
        line = polyfit(map(lambda x:x[1],cands),map(lambda x:x[0],cands),deg=1)
        line = poly1d(line)
        cands_old = cands.copy()
        cands = []
        for x,y in points:
            if abs(x-line(y))<=thresh:
                cands.append((x,y))
        cands = array(cands)
        if len(cands)==0:
            return linefit(points,thresh=thresh+0.0075*threshed.shape[1])
    try:
        return line
    except:
        return None

left = linefit(lefts)
right = linefit(rights)
print "done!"

if not left or not right or right(0)<pagemid+0.2*(right(0)-left(0)) or left(0)>pagemid-0.2*(right(0)-left(0)):
    print "unable to reliably detect text-lines... (weird pageframe) aborting postprocessing!"
    imsave(sys.argv[2],croppedimg)
    sys.exit(0)

print "detecting textlines...",
lines = []
lefts = []
rights = []

def maxval(poly,fr,to,step=1):
    maxval = -1
    for i in range(fr,to,step):
        pval = poly(i)
        if pval>maxval:
            maxval = pval
    return maxval

def minval(poly,fr,to,step=1):
    minval = 1e10
    for i in range(fr,to,step):
        pval = poly(i)
        if pval<minval:
            minval = pval
    return minval

def init_worker_lines(args):
    global objects,leftpoly,rightpoly,threshval,means
    objects,threshed,leftpoly,rightpoly,threshval,cands,n_lines = args
    ccands = arr2d_to_C(cands,C.c_int)
    means = zeros((n_lines,cands.shape[1]))
    cmeans = arr2d_to_C(means,C.c_double)    
    dll = C.CDLL("./libdewarping.so")
    dll['calc_line_means'](ccands,cmeans,cands.shape[0],cands.shape[1],n_lines)
    means = arr2d_from_C(cmeans,means.shape[1],means.shape[0])

def extract_line(i):
    global objects,leftpoly,rightpoly,threshval,means
    left = leftpoly
    right = rightpoly
    o = objects[i]
    pl,pr = getleft(o),getright(o)
    if abs(pl[0]-left(pl[1]))>threshval:
        return None
    if abs(pr[0]-right(pr[1]))>threshval:
        return None
    lmeans = means[i,:]
    lmeans = [(x,lmeans[x]) for x in range(len(lmeans)) if lmeans[x]>0]
    poly = polyfit(map(lambda x:x[0],lmeans),map(lambda x:x[1],lmeans),deg=3)
    poly = poly1d(poly)
    return pl,pr,poly

pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(),initializer=init_worker_lines( \
    (objs,threshed,left,right,threshval,cands,len(objs))))
results = pool.map(extract_line,range(len(objs)))
for r in results:
    if r!=None:
        pl,pr,poly = r
        lefts.append(pl)
        rights.append(pr)
        lines.append(poly)

extension = 0.15*(right[0]-left[0])
left[0] -= extension
right[0] += extension

# shift to page boundaries
valmin = minval(lines[0],lefts[0][0],rights[0][0])
lines[0][0] -= valmin
line = lines[-1]
lines = lines[:-1]
valmax = maxval(line,lefts[-1][0],rights[-1][0])
line[0] += threshed.shape[0]-valmax-1
lines.append(line)
print "done!"

if len(lines)<6:
    print "unable to reliably detect text-lines... (too few) aborting postprocessing!"
    imsave(sys.argv[2],croppedimg)
    sys.exit(0)

def col_at(img,x,y):
    if x<0 or y<0:
        return 0
    xh = ceil(x)
    if xh>=img.shape[1]:
        return 0
    yh = ceil(y)
    if yh>=img.shape[0]:
        return 0
    xl = floor(x)
    yl = floor(y)
    xhd = xh-x
    yhd = yh-y
    xld = x-xl
    yld = y-yl
    return img[yl,xl]*xhd*yhd+img[yl,xh]*xld*yhd+img[yh,xl]*xhd*yld+img[yh,xh]*xld*yld

polymap = dict()
def polylen(poly,valmin,valmax,granularity=10000,target=-1):
    global polymap
    if poly in polymap:
        dists = polymap[poly]
        if target==-1:
            return dists[-1]
        else:
            i = argmin(abs(dists-target))
            pos = valmin+1.*i/(granularity-1)*(valmax-valmin)
            return pos
    points = np.zeros((granularity,2))
    dists = np.zeros(granularity)
    for i in range(granularity):
        pos = valmin+1.0*i/(granularity-1)*(valmax-valmin)
        points[i,:] = np.array([pos,poly(pos)])
        if i>0:
            dists[i] = dists[i-1]+norm(points[i,:]-points[i-1,:])
        if target!=-1 and dists[i]>=target:
            return pos
    polymap[poly] = dists
    return dists[-1]

def polycut(side,tline):
    def f(x):
        return norm(x-array([side(x[1]),tline(x[0])]))
    x,y = fmin(f,x0=zeros(2),disp=0)
    return array((x,y))

def init_worker_dewarp(args):
    global croppedimg,lines,left,right,pxwidth
    global cimg_R,cimg_G,cimg_B,dll
    croppedimg,lines,left,right,pxwidth = args
    cimg_R,cimg_G,cimg_B = colimg_to_C(croppedimg)
    dll = C.CDLL("./libdewarping.so")
    
def dewarp_pair(i): #image,line0,line1,left,right,pxwidth):
    global croppedimg,lines,left,right,pxwidth
    global cimg_R,cimg_G,cimg_B,dll
    image = croppedimg
    line0,line1 = lines[i],lines[i+1]
    # determine where textline cuts left/right border
    pl0 = polycut(left,line0)
    pl1 = polycut(left,line1)
    pr0 = polycut(right,line0)
    pr1 = polycut(right,line1)
    # determine aspect ratio
    ## ll = norm(pl0-pl1)
    ## rl = norm(pr0-pr1)
    ## height = 0.5*(ll+rl)
    tl = polylen(line0,pl0[0],pr0[0])
    bl = polylen(line1,pl1[0],pr1[0])
    width = 0.5*(tl+bl)
    topmid = polylen(line0,pl0[0],pr0[0],target=0.5*tl)
    bottommid = polylen(line1,pl1[0],pr1[0],target=0.5*bl)
    topmid = array([topmid,line0(topmid)])
    bottommid = array([bottommid,line1(bottommid)])
    height = norm(topmid-bottommid)
    pxheight = int(ceil(pxwidth*height/width))
    # sample equidistantly between poly of upper/lower line in required resolution
    rect = zeros((pxheight,pxwidth,3))
    ccol_at = dll['col_at']
    ccol_at.restype = C.c_uint
    polydummy = poly1d(0)
    cpolydummy = poly_to_C(polydummy)
    for x in range(int(pxwidth)):
        xratio = 1.*x/(pxwidth-1)
        xtop = polylen(line0,pl0[0],pr0[0],target=xratio*tl)
        ptop = array((xtop,line0(xtop)))
        xbottom = polylen(line1,pl1[0],pr1[0],target=xratio*bl)
        pbottom = array((xbottom,line1(xbottom)))
        for y in range(pxheight):
            yratio = 1.*y/(pxheight-1)
            ptarget = ptop+yratio*(pbottom-ptop)
            #rect[y,x] = col_at(croppedimg,ptarget[0],ptarget[1])
            ptarget = C.c_double(ptarget[0]),C.c_double(ptarget[1])
            rect[y,x,0] = ccol_at(cimg_R,croppedimg.shape[0],croppedimg.shape[1],ptarget[0],ptarget[1],cpolydummy,1,0)
            rect[y,x,1] = ccol_at(cimg_G,croppedimg.shape[0],croppedimg.shape[1],ptarget[0],ptarget[1],cpolydummy,1,0)
            rect[y,x,2] = ccol_at(cimg_B,croppedimg.shape[0],croppedimg.shape[1],ptarget[0],ptarget[1],cpolydummy,1,0)
    return rect

print "starting dewarping..."
topleft,topright = array(lefts[0]),array(rights[0])
bottomleft,bottomright = array(lefts[-1]),array(rights[-1])
h = 0.5*(norm(topleft-bottomleft)+norm(topright-bottomright))
w = mean([norm(array(lefts[i])-array(rights[i])) for i in range(len(lefts))])
width = w/h*img.shape[0]
height = 0
pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(),initializer=init_worker_dewarp( \
    (croppedimg,lines,left,right,width)))
rects = pool.map(dewarp_pair,range(len(lines)-1))
height = sum([r.shape[0] for r in rects])
print "done!"

dewarped = zeros((height,width,3))
start = 0
for r in rects:
    dewarped[start:start+r.shape[0],:,:] = r
    start += r.shape[0]

tmppic = os.tmpnam()+'.png'
imsave(tmppic,dewarped)
os.system('convert -trim '+tmppic+' '+sys.argv[2])
os.unlink(tmppic)
print "performed final cropping"
print "persisted postprocessed image"
shutil.rmtree(os.environ['MPLCONFIGDIR'])
