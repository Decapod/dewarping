#!/usr/bin/python
import sys
from pylab import *
import ctypes as C

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
    print "poly_to_C"
    testpoly = poly1d([1,2,3.5])
    p = poly_to_C(testpoly)
    print p[0],p[1],p[2]
    print "=================================="
    print "poly_from_C"
    p = poly_from_C(p,3)
    print p
