#!/usr/bin/python

from chelper import *
import ctypes as C
import os
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import shutil
import sys
import warnings
import multiprocessing
from itertools import izip
from pylab import *
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import measurements
from scipy.linalg import norm,inv
from scipy.interpolate import interp1d,interp2d
from scipy.optimize import fmin,fmin_cg,fmin_powell,fmin_bfgs,fmin_l_bfgs_b
from scipy.misc import imsave
import cv
import pyflann as flann
from calib import load
from copy import copy

"""
HARDCODED THINGS - FIXME!
"""
debug = 0                                    # debug mode
scale = 1.00                                 # scale factor for output
n_threads = multiprocessing.cpu_count()      # number of threads to use concurrently (default: #cores)
use_shm = 1
"""
END HARDCODED
"""

def test_imagemagick():
    retval = os.system('convert > /dev/null')
    if retval!=0:
        print "ERROR: imagemagick not installed"
        sys.exit(2)

def test_sift():
    retval = os.system('sift > /dev/null')
    if retval!=0:
        print "ERROR: vlfeat executable 'sift' not in path"
        sys.exit(3)

def usage():
    print "WARNING: There are still hard-coded parts in the script, which have to be changed!"
    print "debug       -> enables or disables warnings"
    print "scale       -> defines output scale factor, used for speeding up test runs"
    print "n_threads   -> number of threads to use concurrently, default=cores-1"
    print
    print "Decapod - Stereo Dewarping"
    print sys.argv[0],"<dir_calib> <imgl> <imgr> <dewarped_out> [--only3d]"
    print "To disable postprocessing append --only3d to the above command."
    print
    print "Performs stereo dewarping of image pair."
    print "  dir_calib:     directory to read calibration matrices from"
    print "  imgl:          path to left image of stereo pair"
    print "  imgr:          path to right image of stereo pair"
    print "  dewarped_out:  directory to persist dewarped images (and debugging output) to"
    sys.exit(1)

try:
    f = open(".colors",'rb')
    data = f.readlines()
    sepcol = data[0]
    bgcol = data[1]
    sepcol_r,sepcol_g,sepcol_b = map(int,sepcol.split())
    bgcol_r,bgcol_g,bgcol_b = map(int,bgcol.split())
except:
    print "ERROR: You have to run colsel.py first to select page separator and background color!"
    sys.exit(2)

warnings.filterwarnings('ignore')
test_imagemagick()
test_sift()
if len(sys.argv)>5:
    if sys.argv[5]=="--only3d":
        print "Disabled postprocessing!"
        postproc = 0
        both = 0
    elif sys.argv[5]=="--both":
        print "Outputting 3d dewarping and postprocessing!"
        postproc = 1
        both = 1
    else:
        usage()
elif len(sys.argv)==5:
    postproc = 1
    both = 0
else:
    usage()

if scale!=1.00:
    print "WARNING: output scaled by "+str(scale)

def sift(imgpath,img):
    def isbg(img,x,y):
        xf,xc,yf,yc = int(floor(x)),int(ceil(x)),int(floor(y)),int(ceil(y))
        xf = max(0,xf)
        xc = max(0,xc)
        yf = max(0,yf)
        yc = max(0,yc)
        h,w = img.shape
        xf = min(xf,w-1)
        xc = min(xc,w-1)
        yf = min(yf,h-1)
        yc = min(yc,h-1)
        if img[yf,xf]==0:
            return 1
        if img[yc,xf]==0:
            return 1
        if img[yf,xc]==0:
            return 1
        if img[yc,xc]==0:
            return 1
        return 0
    if not use_shm:
        pgmout = os.tmpnam()+".pgm"
        featsout = os.tmpnam()
    else:
        pgmout = '/dev/shm/'+os.tmpnam().split('/')[-1]+'.pgm'
        featsout =  '/dev/shm/'+os.tmpnam().split('/')[-1]
    os.system("convert "+imgpath+" "+pgmout)
    os.system("sift "+pgmout+" -o "+featsout)
    os.unlink(pgmout)
    featsfile = open(featsout,'rb')
    data = featsfile.readlines()
    featsfile.close()
    os.unlink(featsout)
    f = zeros((2,len(data)))
    d = zeros((len(data),len(data[0].split())-4))
    for i,line in enumerate(data):
        f[:,i] = map(float,line.split()[:2])
        d[i,:] = map(int,line.split()[4:])
    # remove features from background
    goods = []
    for i in range(f.shape[1]):
        if isbg(img,f[0,i],f[1,i]):
            continue
        goods.append(i)
    f = f[:,goods]
    d = d[goods,:]
    return f,d

def match(x):
    fl,dl,fr,dr = x
    def __match(dl,nnl,dr,nnr):
        matches = []
        for i in range(dl.shape[0]):
            idx,dist = nnr.nn_index(dl[i,:],num_neighbors=2)
            # ensure distinctiveness
            if dist[0][0]/dist[0][1] > 0.8:
                continue
            # ensure uniqueness
            index = idx[0][0]
            idx,dist = nnl.nn_index(dr[index,:],num_neighbors=1)
            if idx[0]!=i:
                continue
            # good match!
            matches.append((i,index))
        return matches
    def __ransac(fl,fr,matches):
        n = len(matches)
        fl_cv = cv.CreateMat(2,n,cv.CV_64F)
        fr_cv = cv.CreateMat(2,n,cv.CV_64F)
        for i in range(n):
            fl_cv[0,i] = fl[0,matches[i][0]]
            fl_cv[1,i] = fl[1,matches[i][0]]
            fr_cv[0,i] = fr[0,matches[i][1]]
            fr_cv[1,i] = fr[1,matches[i][1]]
        F = cv.CreateMat(3,3,cv.CV_64F)
        status = cv.CreateMat(1,n,cv.CV_8U)
        cv.FindFundamentalMat(fl_cv,fr_cv,F,cv.CV_FM_RANSAC,5.0,0.9999,status)
        matches_cleaned = []
        for i in range(n):
            if status[0,i] == 1:
                matches_cleaned.append(matches[i])
        return matches_cleaned
    
    nnl = flann.FLANN()
    nnl.build_index(dl,num_neighbors=2,target_precision=0.9999)
    nnr = flann.FLANN()
    nnr.build_index(dr,num_neighbors=2,target_precision=0.9999)
    matches = __match(dl,nnl,dr,nnr)
    matches = __ransac(fl,fr,matches)
    return matches

def reconstruct3d(imgl_path,imgr_path,imgl,imgr):
    # sift takes quite a bit of memory, ensure we have at least 8GB to run in parallel
    freeram = int(os.popen("free -m").readlines()[1].split()[1])
    if freeram>8000:
        nproc = 2
    else:
        nproc = 1
    pool = multiprocessing.Pool(processes=nproc)
    res_l = pool.apply_async(sift,(imgl_path,imgl))
    res_r = pool.apply_async(sift,(imgr_path,imgr))
    pool.close()
    pool.join()
    fl,dl = res_l.get()
    fr,dr = res_r.get()
    pool = multiprocessing.Pool(processes=n_threads)
    matches = pool.map(match,[(fl,dl,fr,dr) for i in range(n_threads)])
    matches.sort(key=lambda x:len(x),reverse=1)
    return fl,fr,matches[0]

def pagesep(img,maxdist=5,r=sepcol_r,g=sepcol_g,b=sepcol_b):
    def find_col(img,col,thresh=0.25):
        img = img.copy()
        if amax(img)>1:
            img /= amax(img)
        img -= col
        img = sqrt(sum(img**2,axis=2))
        img -= amin(img)
        img /= amax(img)
        img = where(img>thresh,0,1)
        return img
    # find page separator by color similarity
    stick = find_col(img,array([r/255.,g/255.,b/255.]))
    counts = dict()
    for x in range(img.shape[1]):
        num = 0
        for y in range(img.shape[0]):
            if stick[y,x]==1:
                num += 1
        counts[x] = num
    # find feasible candidates
    candidates = []
    n_iter,multiplier = 0,0.15
    while n_iter<5 and len(candidates)==0:
        minpixels = multiplier*img.shape[0]
        candidates = []
        for key in counts.keys():
            if counts[key]>minpixels:
                candidates.append(key)
        n_iter += 1
        multiplier *= 0.5
    nearmid = array(candidates)-0.5*img.shape[1]
    index = argmin(abs(nearmid))
    mid = candidates[index]
    #lower,upper = mid-0.5*img.shape[1],mid+0.5*img.shape[1]   #0.2
    #stick[:,:lower] = 0
    #stick[:,upper+1:] = 0
    if debug:
        imsave(sys.argv[4]+"/stick.png",stick)
    # find horizontal mean of color per line
    means = []
    septop,sepbottom = 10000,-1
    for y in range(stick.shape[0]):
        ones = []
        for x in range(stick.shape[1]):
            if stick[y,x]==1:
                ones.append(x)
        if len(ones)>0:
            if y<septop:
                septop = y
            if y>sepbottom:
                sepbottom = y
            means.append((y,mean(ones)))
    # rlse
    means_work,means_old = copy(means),[]
    while means_old!=means_work:
        means_old = copy(means_work)
        p = polyfit(map(lambda x:x[0],means_work),map(lambda x:x[1],means_work),deg=1)
        p = poly1d(p)
        means_work = []
        for (y,x) in means:
            ppoly = ((y,p(y)))
            if norm(array((y,x))-array(ppoly))<=maxdist:
                means_work.append((y,x))
    return p,septop,sepbottom

def pagesep_3d(imgl,imgr):
    poly_l,p_l_top,p_l_bottom = pagesep(imgl)
    poly_r,p_r_top,p_r_bottom = pagesep(imgr)
    p_top = array([poly_l(p_l_top),p_l_top,poly_l(p_l_top)-poly_r(p_r_top)])
    p_bottom = array([poly_l(p_l_bottom),p_l_bottom,poly_l(p_l_bottom)-poly_r(p_r_bottom)])
    area_top = area(poly_l,min(p_l_top,p_l_bottom),max(p_l_top,p_l_bottom))
    area_bottom = area(poly_r,min(p_r_top,p_r_bottom),max(p_r_top,p_r_bottom))
    return p_top,p_bottom,poly_l,poly_r,0.5*(area_top+area_bottom)

def transform_hom(ps,mat):
    hom = ones((ps.shape[0],4))
    hom[:,:3] = ps
    trans = dot(mat,hom.T).T
    trans = trans[:,:3]/trans[:,3].repeat(3).reshape(-1,3)
    return trans

# Cleaning with some hacked heuristics and initial surface fit.
def dist_poly_point(p,poly,rng,samples=100):
    xs = linspace(rng[0],rng[1],samples)
    ys = map(poly,xs)
    xys = array(zip(xs,ys))
    dists = xys-p
    dists = array(map(norm,dists))
    return amin(dists)

def naive_clean(points):
    goods = zeros(points.shape[0],dtype='bool')
    center = mean(points[:,2])
    dists = map(norm,points[:,2]-center)
    mu = mean(dists)
    sigma = std(dists)
    for i in range(points.shape[0]):
        if dists[i]<=mu+3*sigma:
            goods[i] = 1
    xmean = mean(points[:,0])
    dists = map(norm,points[:,0]-xmean)
    mu = mean(dists)
    sigma = std(dists)
    for i in range(points.shape[0]):
        if goods[i]==1 and dists[i]<=mu+3*sigma:
            goods[i] = 1
    ymin,ymax = amin(points[:,0]),amax(points[:,0])
    ys,zs = points[:,0],points[:,2]
    poly = poly1d(polyfit(ys,zs,2))
    dists = zeros(points.shape[0])
    for i in range(points.shape[0]):
        dists[i] = abs(points[i,2]-poly(points[i,0])) #dist_poly_point(points[i,1:],poly,(ymin,ymax))
    mu = mean(dists)
    sigma = std(dists)
    for i in range(points.shape[0]):
        if goods[i]==1 and dists[i]<=mu+2.75*sigma:
            goods[i] = 1
        else:
            goods[i] = 0
    return goods

def rot(x,y,z):
    Rinv = zeros((4,4))
    Rinv[:3,:3] = array([x,y,z]).T
    Rinv[3,3] = 1
    return inv(Rinv)

def rotmat(p,lp0,lp1,angle):
    u = lp1-lp0
    u /= norm(u)
    a,b,c = lp0
    u,v,w = u
    x,y,z = p
    C = cos(angle)
    Ci = 1-C
    S = sin(angle)
    newp = array([ \
      (a*(v*v+w*w)-u*(b*v+c*w-u*x-v*y-w*z))*Ci+x*C+(-c*v+b*w-w*y+v*z)*S, \
      (b*(u*u+w*w)-v*(a*u+c*w-u*x-v*y-w*z))*Ci+y*C+(c*u-a*w+w*x-u*z)*S, \
      (c*(u*u+v*v)-w*(a*u+b*v-u*x-v*y-w*z))*Ci+z*C+(-b*u+a*v-v*x+u*y)*S])
    return newp
    
def nextp_line(lp0,lp1,p):
    pl = p-lp0
    ll = lp1-lp0
    t = dot(pl,ll)/norm(ll)**2
    return array(lp0+t*ll)

def remove_bg(img,bgcol,sigma=25,thresh=50):
    smoothed = gaussian_filter(img,(sigma,sigma,0))
    sub = abs(smoothed-bgcol)
    subsum = mean(sub,axis=2)
    bg = where(subsum<thresh,1,0)
    img = img.copy()
    img[bg==1,:] = 0
    gimg = mean(img,axis=2)
    nonblack = gimg!=0
    labeled,n = measurements.label(nonblack)
    biggest,nbiggest = -1,-1
    for i in range(1,n+1):
        npix = sum(labeled==i)
        if npix>nbiggest:
            biggest = i
            nbiggest = npix
    for i in range(1,n+1):
        if i!=biggest:
            img[labeled==i] = zeros(3)
    return img

def col_at(img,x,y,poly_sep,direction):
    if x<0 or y<0:
        return 0
    xh = ceil(x)
    if xh>=img.shape[1]:
        return 0
    yh = ceil(y)
    if yh>=img.shape[0]:
        return 0
    if direction!=(x<poly_sep(y)):
        return 0
    xl = floor(x)
    yl = floor(y)
    xhd = xh-x
    yhd = yh-y
    xld = x-xl
    yld = y-yl
    return img[yl,xl]*xhd*yhd+img[yl,xh]*xld*yhd+img[yh,xl]*xhd*yld+img[yh,xh]*xld*yld

def dist_point_plane(p,a,b,c):
    n = cross(b-a,c-a)
    n /= norm(n)
    d = dot(n,p-a)
    return d

def area(poly,fr,to):
    return polylen(poly,fr,to)

polymap = dict()
def polylen(poly,valmin,valmax,granularity=100000,target=-1):
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

def findpos(poly,parea,ratio,fr,xmax):
    return polylen(poly,fr,xmax,target=ratio*parea)

def process_col(i):
    global steps,polyl,al,polyu,au,xmin,xmax,ymin,ymax,R2inv,Rinv,Qinv,height,poly_sep,direction,imgl_col,mattrans
    global cpoly_sep,cpoly_sep_order,cimgl_R,cimgl_G,cimgl_B,dll
    ratio = i*1./(steps-1)
    targetl = findpos(polyl,al,ratio,xmin,xmax)
    targetu = findpos(polyu,au,ratio,xmin,xmax)
    p0 = array([targetl,ymin,polyl(targetl)])
    p1 = array([targetu,ymax,polyu(targetu)])
    points = array([p0,p1])
    pback = transform_hom(points,R2inv)
    pback = transform_hom(pback,Rinv)
    pback = transform_hom(pback,Qinv)
    #pback = transform_hom(points,mattrans)
    p0,p1 = pback
    v = p1-p0
    ccol_at = dll['col_at']
    ccol_at.restype = C.c_uint
    column = zeros((height,1,3))
    for j in range(height):
        ratio = j*1./(height-1)
        pos = p0+ratio*v
        #column[j,0,:] = col_at(imgl_col,pos[0],pos[1],poly_sep,direction)
        pos = C.c_double(pos[0]),C.c_double(pos[1])
        column[j,0,0] = ccol_at(cimgl_R,imgl_col.shape[0],imgl_col.shape[1],pos[0],pos[1],cpoly_sep,cpoly_sep_order,direction)
        column[j,0,1] = ccol_at(cimgl_G,imgl_col.shape[0],imgl_col.shape[1],pos[0],pos[1],cpoly_sep,cpoly_sep_order,direction)
        column[j,0,2] = ccol_at(cimgl_B,imgl_col.shape[0],imgl_col.shape[1],pos[0],pos[1],cpoly_sep,cpoly_sep_order,direction)
    return column

def init_worker(args):
    global steps,polyl,al,polyu,au,xmin,xmax,ymin,ymax,R2inv,Rinv,Qinv,height,poly_sep,direction,imgl_col,mattrans
    global cpoly_sep,cpoly_sep_order,cimgl_R,cimgl_G,cimgl_B,dll
    steps,polyl,al,polyu,au,xmin,xmax,ymin,ymax,R2inv,Rinv,Qinv,height,poly_sep,direction,imgl_col,mattrans = args
    cpoly_sep = poly_to_C(poly_sep)
    cpoly_sep_order = poly_sep.order+1
    cimgl_R,cimgl_G,cimgl_B = colimg_to_C(imgl_col)
    dll = C.CDLL("./libdewarping.so")

def dewarp_page(points,outpath,poly_sep,direction,no_run,area_sep,maxratio=2.5,thresh=0.85,n_outer=75,degree=2):
    def best(q):
        def fit(alpha):
            p = rotmat(q,line0[0],line0[1],alpha)
            dists = []
            for point in p3d_book:
                dists.append(dist_point_plane(point,p,line0[0],line0[1]))
            dists = array(dists)
            return sum(dists)
        alpha = fmin_l_bfgs_b(fit,zeros(1),approx_grad=1,bounds=[(-radians(180),radians(180))])[0]
        bestp = rotmat(q,line0[0],line0[1],alpha)
        return array([bestp[0][0],bestp[1][0],bestp[2][0]])
    
    p3d = transform_hom(points,Q)
    goods = naive_clean(p3d)
    p3d_clean = p3d[goods]

    # Orientation normalization by plane fitting.
    center = mean(p3d_clean,axis=0)
    A = dot((p3d_clean-center).T,(p3d_clean-center))
    w,v = eig(A)
    minev = argmin(w)
    axisz = v[:,minev]
    axisy = -cross(array([1,0,0]),axisz)
    axisx = cross(axisy,axisz)
    axisx /= norm(axisx)
    axisy /= norm(axisy)
    axisz /= norm(axisz)

    R = rot(axisx,axisy,axisz)
    Rinv = inv(R)
    p3d_book = transform_hom(p3d_clean,R)

    xmin,xmax = amin(p3d_book[:,0]),amax(p3d_book[:,0])
    ymin,ymax = amin(p3d_book[:,1]),amax(p3d_book[:,1])
    zmin,zmax = amin(p3d_book[:,2]),amax(p3d_book[:,2])

    # Should be possible to get better results with a proper seed line.
    line0 = array([l0bottom,l0top])
    line0 = transform_hom(line0,Q)
    line0 = transform_hom(line0,R)

    # Let's finally try to fit a polynomial. This should be smoother.
    # First: estimate best plane fit given seed line
    q = line0[0].copy()
    if direction:
        q[0] += 10
    else:
        q[0] -= 10

    q = best(q)

    axis2 = line0[0]-line0[1]
    axis2 /= -norm(axis2)

    axis1 = nextp_line(line0[0],line0[1],q)-q
    axis1 /= -norm(axis1)

    axis3 = cross(axis1,axis2)
    axis3 /= norm(axis3)

    #R2 = rot(axis1,axis2,axis3)
    if direction:
        R2 = array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype='float')
    else:
        R2 = array([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]],dtype='float')
    R2inv = inv(R2)
    p3d_book2 = transform_hom(p3d_book,R2)
    line0_2 = transform_hom(line0,R2)
    mattrans = dot(dot(R2inv,Rinv),Qinv)

    # handle outer slices more robustly by fitting a plane and only fit higher order poly to dense regions
    xs = p3d_book2[:,0]
    order = argsort(xs)
    p3d_book2 = p3d_book2[order]
    outer_left = p3d_book2[:n_outer,:]
    outer_right = p3d_book2[-n_outer:,:]
    if direction:
        xridge = p3d_book2[0,0]
    else:
        xridge = p3d_book2[-1,0]

    xmin,xmax = amin(p3d_book2[:,0]),amax(p3d_book2[:,0])
    ymin,ymax = amin(p3d_book2[:,1]),amax(p3d_book2[:,1])
    rng = ymax-ymin
    ymin -= 0.15*rng
    ymax += 0.15*rng
    ymid = ymin+0.5*(ymax-ymin)

    uppers_orig = array([(x,y,z) for (x,y,z) in p3d_book2 if y>=ymid])#,line0_2,axis=0)
    lowers_orig = array([(x,y,z) for (x,y,z) in p3d_book2 if y<=ymid])#,line0_2,axis=0)
    uppers_old = uppers_orig.copy()
    lowers_old = lowers_orig.copy()

    if uppers_orig.shape[0]<500 or lowers_orig.shape[0]<500:
        print "WARNING: not enough matches given by FLANN... re-executing!"
        return 0

    points_orig = p3d_book2.copy()
    uppers_orig = points_orig.copy()
    uppers_old = uppers_orig.copy()
    
    while True:
        print "relevant points:",uppers_old.shape[0],"of",uppers_orig.shape[0]
        polyu = polyfit(uppers_old[:,0],uppers_old[:,2],deg=degree)
        polyu = poly1d(polyu)
        uppers = []
        for (x,y,z) in uppers_orig:
            ppoly = array((x,y,polyu(x)))
            dist = norm(ppoly-array((x,y,z)))
            if dist<thresh:
                uppers.append((x,y,z))
        uppers = array(uppers)
        #uppers = append(uppers,line0_2,axis=0)
        if uppers.shape[0]==uppers_old.shape[0]:
            good = 1
            for i in range(uppers.shape[0]):
                if (uppers[i,:]!=uppers_old[i,:]).any():
                    good = 0
                    break
            if good:
                break
        uppers_old = uppers.copy()
    print "fitted upper poly"
    ## while True:
    ##     print "relevant points:",lowers_old.shape[0],"of",lowers_orig.shape[0]
    ##     polyl = polyfit(lowers_old[:,0],lowers_old[:,2],deg=degree)
    ##     polyl = poly1d(polyl)
    ##     lowers = []
    ##     for (x,y,z) in lowers_orig:
    ##         ppoly = array((x,y,polyl(x)))
    ##         dist = norm(ppoly-array((x,y,z)))
    ##         if dist<thresh:
    ##             lowers.append((x,y,z))
    ##     lowers = array(lowers)
    ##     #lowers = append(lowers,line0_2,axis=0)
    ##     if lowers.shape[0]==lowers_old.shape[0]:
    ##         good = 1
    ##         for i in range(lowers.shape[0]):
    ##             if (lowers[i,:]!=lowers_old[i,:]).any():
    ##                 good = 0
    ##                 break
    ##         if good:
    ##             break
    ##     lowers_old = lowers.copy()
    ## print "fitted lower poly"

    polyu = poly1d(polyfit(uppers[:,0],uppers[:,2],deg=5))
    #polyl = poly1d(polyfit(lowers[:,0],lowers[:,2],deg=3))
    
    polyl = polyu
    lowers = uppers.copy()

    uxmin,uxmax = amin(uppers[:,0]),amax(uppers[:,0])
    lxmin,lxmax = amin(lowers[:,0]),amax(lowers[:,0])
    xmin = min(uxmin,lxmin)
    xmax = max(uxmax,lxmax)
    rng = xmax-xmin
    xmin -= 0.15*rng
    xmax += 0.15*rng

    au = area(polyu,xmin,xmax)
    al = area(polyl,xmin,xmax)
    a = 0.5*(au+al)

    height_len = 1.3*norm(line0_2[0]-line0_2[1])
    width_len = a
    ratio = width_len/height_len
    width = int(ceil(ratio*height))
    print "output size = "+str(width)+" x "+str(height)

    if debug:
        xp = linspace(xmin,xmax,100)
        plot(-p3d_book2[:,0],p3d_book2[:,2],'go')
        plot(-uppers[:,0],uppers[:,2],'ro')
        plot(-lowers[:,0],lowers[:,2],'bo')
        plot(-xp,polyu(xp),'r-',-xp,polyl(xp),'b-',linewidth=2.5)
        fname = sys.argv[4]+"/poly"+str(no_run)+".png"
        savefig(fname,dpi=320)
        clf()

    if ratio>maxratio or ratio<(1.0/maxratio):
        print "WARNING: bad polynomial fit... re-executing!"
        return 0

    dewarped = zeros((height,width,3))
    steps_full = width
    steps = steps_full

    pool = multiprocessing.Pool(processes=n_threads,initializer=init_worker( \
        (steps,polyl,al,polyu,au,xmin,xmax,ymin,ymax,R2inv,Rinv,Qinv,height,poly_sep,direction,imgl_col,mattrans)))
    columns = pool.map(process_col,range(steps))
    for i in range(len(columns)):
        dewarped[:,i,:] = columns[i][:,0,:]
    
    # normalize orientation
    topleft = array([xmin,ymin,polyl(xmin)])
    topright = array([xmax,ymin,polyl(xmax)])
    bottomleft = array([xmin,ymax,polyl(xmin)])
    bottomright = array([xmax,ymax,polyl(xmax)])
    points = array([topleft,topright,bottomleft,bottomright])
    points = transform_hom(points,R2inv)
    points = transform_hom(points,Rinv)
    points = transform_hom(points,Qinv)
    topleft,topright,bottomleft,bottomright = points
    if topleft[0]>topright[0]:
        dewarped = fliplr(dewarped)
    if topleft[1]>bottomleft[1]:
        dewarped = flipud(dewarped)

    # crop
    gimg = mean(dewarped,axis=2)
    mask = gimg!=0
    rowsums = array([sum(mask[i,:]) for i in range(mask.shape[0])])
    ynonzero = [i for i in range(rowsums.shape[0]) if rowsums[i]>0]
    ymin,ymax = min(ynonzero),max(ynonzero)
    colsums = array([sum(mask[:,i]) for i in range(mask.shape[1])])
    xnonzero = [i for i in range(colsums.shape[0]) if colsums[i]>0]
    xmin,xmax = min(xnonzero),max(xnonzero)
    cropped = dewarped[ymin:ymax+1,xmin:xmax+1,:]
            
    imsave(outpath,cropped)
    return 1

"""
MAIN
"""
try:
    if debug:
        os.mkdir(sys.argv[4])
except:
    pass

print "Check whether C module is compiled and linked..."
if not os.path.exists("./libdewarping.so"):
    os.system("scons")

print "Loading calibration data..."
Cl,Cr,dl,dr,R,T,F,E,Pl,Pr,Rl,Rr,Q = load(sys.argv[1])

print "Rectifying stereo images..."
imgl = cv.LoadImage(sys.argv[2])
imgr = cv.LoadImage(sys.argv[3])
mapl1 = cv.CreateMat(imgl.height,imgl.width,cv.CV_32FC1)
mapl2 = cv.CreateMat(imgl.height,imgl.width,cv.CV_32FC1)
mapr1 = cv.CreateMat(imgl.height,imgl.width,cv.CV_32FC1)
mapr2 = cv.CreateMat(imgl.height,imgl.width,cv.CV_32FC1)
cv.InitUndistortRectifyMap(Cl,dl,Rl,Pl,mapl1,mapl2)
cv.InitUndistortRectifyMap(Cr,dr,Rr,Pr,mapr1,mapr2)
rectl = cv.CreateImage(cv.GetSize(imgl),imgl.depth,imgl.nChannels)
rectr = cv.CreateImage(cv.GetSize(imgl),imgl.depth,imgl.nChannels)
cv.Remap(imgl,rectl,mapl1,mapl2)
cv.Remap(imgr,rectr,mapr1,mapr2)
if debug:
    cv.SaveImage(sys.argv[4]+"/rectified_left.png",rectl)
    cv.SaveImage(sys.argv[4]+"/rectified_right.png",rectr)
else:
    lpath = os.tmpnam()+".png"
    rpath = os.tmpnam()+".png"
    cv.SaveImage(lpath,rectl)
    cv.SaveImage(rpath,rectr)

print "Loading stereo image..."
if debug:
    imgl = array(imread(sys.argv[4]+"/rectified_left.png"),dtype='float32')
    imgr = array(imread(sys.argv[4]+"/rectified_right.png"),dtype='float32')
else:
    imgl = array(imread(lpath),dtype='float32')
    imgr = array(imread(rpath),dtype='float32')
    
imgl *= 255.
imgr *= 255.
# fix output height to image height and derive width based on shape later on
height = int(floor(imgl.shape[0]*scale))

print "Removing background..."
bgcol = array([bgcol_r,bgcol_g,bgcol_b])
imgl = remove_bg(imgl,bgcol)
imgr = remove_bg(imgr,bgcol)
if debug:
    imsave(sys.argv[4]+"/cleaned_left.png",imgl)
    imsave(sys.argv[4]+"/cleaned_right.png",imgr)

print "Detecting page separator..."
l0top,l0bottom,poly_l,poly_r,area_sep = pagesep_3d(imgl,imgr)

imgl_col = imgl.copy()
imgl = mean(imgl,axis=2)
imgr = mean(imgr,axis=2)

if debug:
    tmpleft = sys.argv[4]+"/dewarped-left.png"
    tmpright = sys.argv[4]+"/dewarped-right.png"
else:
    if both:
        tmpleft = sys.argv[4]+"-dewarped-left.png"   #os.tmpnam()+".png"
        tmpright = sys.argv[4]+"-dewarped-right.png" #os.tmpnam()+".png"
    else:
        tmpleft = os.tmpnam()+".png"
        tmpright = os.tmpnam()+".png"

run = 0
dewarped_left,dewarped_right = 0,0
while dewarped_left==0 or dewarped_right==0:
    print "Performing stereo matching..."
    fl,fr,matches = reconstruct3d(sys.argv[2],sys.argv[3],imgl,imgr)

    print "Determining disparities of feature matches..."
    pdisp = zeros((len(matches),3))
    pdisp[:,:2] = fl[:2,map(lambda x:x[0],matches)].T
    pdisp[:,2] = pdisp[:,0]-fr[0,map(lambda x:x[1],matches)]

    print "Splitting left and right page..."
    pleft,pright = [],[]
    for i in range(pdisp.shape[0]):
        xsep = poly_l(pdisp[i,1])
        if pdisp[i,0]<=xsep:    
            pleft.append(pdisp[i,:])
        else:
            pright.append(pdisp[i,:])
    pright,pleft = array(pright),array(pleft)
    
    print "Transforming to real depth..."
    Q = asarray(Q)
    Qinv = inv(Q)

    if not dewarped_left:
        print "Dewarping left page..."
        if debug:
            imgout_l = sys.argv[4]+"/postprocessed-left.png"
        else:
            if both:
                imgout_l = sys.argv[4]+"-postprocessed-left.png"
            else:
                imgout_l = sys.argv[4]+"-left.png"
        if postproc:
            ret = dewarp_page(pleft,tmpleft,poly_l,1,run,area_sep)
        else:
            ret = dewarp_page(pleft,imgout_l,poly_l,1,run,area_sep)
        run += 1
        if ret:
            dewarped_left = 1

    if not dewarped_right:
        print "Dewarping right page..."
        if debug:
            imgout_r = sys.argv[4]+"/postprocessed-right.png"
        else:
            if both:
                imgout_r = sys.argv[4]+"-postprocessed-right.png"
            else:
                imgout_r = sys.argv[4]+"-right.png"
        if postproc:
            ret = dewarp_page(pright,tmpright,poly_l,0,run,area_sep)
        else:
            ret = dewarp_page(pright,imgout_r,poly_l,0,run,area_sep)
        run += 1
        if ret:
            dewarped_right = 1

    if run>10:
        print "ERROR: Could not dewarp images!"
        sys.exit(10)

if postproc:
    print "Starting postprocessing of left page..."
    os.system("./postproc.py "+tmpleft+" "+imgout_l)
    print "...done!"
    print "Starting postprocessing of right page..."
    os.system("./postproc.py "+tmpright+" "+imgout_r)
    print "...done!"

if not debug and not both and postproc:
    os.unlink(tmpleft)
    os.unlink(tmpright)
if not debug:
    os.unlink(lpath)
    os.unlink(rpath)

shutil.rmtree(os.environ['MPLCONFIGDIR'])
print "Finished!"
