#!/usr/bin/python
import sys
from pylab import *
import ctypes as C

def arr2d_to_C(arr,type):
    h,w = arr.shape
    carr_type = type*(h*w)
    carr = carr_type()
    for i in range(w):
        for j in range(h):
            carr[j*w+i] = arr[j,i]
    return carr

def arr2d_from_C(carr,w,h):
    arr = zeros((h,w))
    for i in range(w):
        for j in range(h):
            arr[j,i] = carr[j*w+i]
    return arr

def img_to_C(img):
    img -= amin(img)
    img *= 255./amax(img)
    h,w = img.shape[:2]
    n = h*w
    cimg_type = C.c_uint*n
    cimg = cimg_type()
    for x in range(w):
        for y in range(h):
            cimg[y*w+x] = img[y,x]
    return cimg

def img_from_C(cimg,w,h):
    img = zeros((h,w))
    for x in range(w):
        for y in range(h):
            img[y,x] = cimg[y*w+x]
    return img

def colimg_to_C(img):
    img -= amin(img)
    img *= 255./amax(img)
    h,w = img.shape[:2]
    n = h*w
    cimg_type = C.c_uint*n
    cimg_R = cimg_type()
    cimg_G = cimg_type()
    cimg_B = cimg_type()
    for x in range(w):
        for y in range(h):
            cimg_R[y*w+x] = img[y,x,0]
            cimg_G[y*w+x] = img[y,x,1]
            cimg_B[y*w+x] = img[y,x,2]
    return cimg_R,cimg_G,cimg_B

def colimg_from_C(cimg_R,cimg_G,cimg_B,w,h):
    img = zeros((h,w,3))
    for x in range(w):
        for y in range(h):
            img[y,x,0] = cimg_R[y*w+x]
            img[y,x,1] = cimg_G[y*w+x]
            img[y,x,2] = cimg_B[y*w+x]
    return img

def poly_to_C(poly):
    n = poly.order+1
    poly_type = C.c_double*n
    cpoly = poly_type()
    for i,c in enumerate(poly.c):
        cpoly[n-i-1] = c
    return cpoly

def poly_from_C(cpoly,n):
    poly = zeros(n)
    for i in range(n):
        poly[n-i-1] = cpoly[i]
    poly = poly1d(poly)
    return poly

if sys.argv[0]=="./chelper.py":
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

    lib = C.CDLL("./libdewarping.so")
    """
    from scipy.misc import imsave
    print "colimg_to_C"
    testimg = imread("left.png")
    iR,iG,iB = colimg_to_C(testimg)
    print iR[311221]
    print "=================================="
    print "colimg_from_C"
    img = colimg_from_C(iR,iG,iB,testimg.shape[1],testimg.shape[0])
    imsave("testcolimg_from_C.png",img)
    print "=================================="
    """
    print "poly_to_C"
    testpoly = poly1d([1,2,3.5])
    cp = poly_to_C(testpoly)
    print cp[2],cp[1],cp[0]
    print "=================================="
    print "poly_from_C"
    p = poly_from_C(cp,3)
    print p
    print "=================================="
    print "=================================="
    print "poly evaluation in C"
    print "poly(5.1) =",p(5.1)
    eval_poly = lib.eval_poly
    eval_poly.restype = C.c_double
    print "cpoly(5.1) =",eval_poly(cp,3,C.c_double(5.1))
    print "=================================="
    print "interpolation in C"
    testimg = imread("left.png")
    iR,iG,iB = colimg_to_C(testimg)
    print "col_at(img,101.1,101.1,0,0) =",col_at(testimg,101.75,101.75,poly1d(0),0)
    col_at = lib.col_at
    col_at.restype = C.c_uint
    valR = col_at(iR,testimg.shape[0],testimg.shape[1],C.c_double(101.75),C.c_double(101.75),poly_to_C(poly1d(0)),1,0)
    valG = col_at(iG,testimg.shape[0],testimg.shape[1],C.c_double(101.75),C.c_double(101.75),poly_to_C(poly1d(0)),1,0)
    valB = col_at(iB,testimg.shape[0],testimg.shape[1],C.c_double(101.75),C.c_double(101.75),poly_to_C(poly1d(0)),1,0)
    print "ccol_at(img,101.1,101.1,0,0) =",valR
    print "ccol_at(img,101.1,101.1,0,0) =",valG
    print "ccol_at(img,101.1,101.1,0,0) =",valB
