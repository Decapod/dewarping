import math
import numpy as np
import cv

__p_corner = cv.CV_CALIB_CB_NORMALIZE_IMAGE+cv.CV_CALIB_CB_ADAPTIVE_THRESH+cv.CV_CALIB_CB_FILTER_QUADS
__p_cam = cv.CV_CALIB_FIX_PRINCIPAL_POINT
__p_rig = cv.CV_CALIB_FIX_INTRINSIC

def __corners(img,board_size,params=__p_corner):
    found,corners = cv.FindChessboardCorners(img,board_size,params)
    if not found or len(corners)!=np.prod(board_size):
        raise "could not locate chessboard corners"
    size = cv.GetSize(img)
    img_gray = cv.CreateImage(size,8,1)
    cv.CvtColor(img,img_gray,cv.CV_BGR2GRAY)
    cv.FindCornerSubPix(img_gray,corners,(5,5),(-1,-1),(cv.CV_TERMCRIT_EPS+cv.CV_TERMCRIT_ITER,100,1e-4))
    return corners

def __detect(imgpairs,board_size):
    corner_pairs = []
    for l,r in imgpairs:
        try:
            corners_l = __corners(l,board_size)
            corners_r = __corners(r,board_size)
            corner_pairs.append((corners_l,corners_r))
        except:
            pass
    if len(corner_pairs)==0:
        raise "could not locate chessboard corners in any of the input images"
    return corner_pairs

def __points(corner_pairs,board_size):
    n_good = len(corner_pairs)
    board_n = np.prod(board_size)
    n_points = n_good*board_n
    pl = cv.CreateMat(n_points,1,cv.CV_32FC2)
    pr = cv.CreateMat(n_points,1,cv.CV_32FC2)
    pobj = cv.CreateMat(n_points,3,cv.CV_32FC1)
    pn = cv.CreateMat(n_good,1,cv.CV_32SC1)
    for i,(l,r) in enumerate(corner_pairs):
        for j in range(board_n):
            idx = i*board_n+j
            pl[idx,0] = (l[j][0],l[j][1])
            pr[idx,0] = (r[j][0],r[j][1])
            pobj[idx,0] = j/board_size[0]
            pobj[idx,1] = j%board_size[0]
            pobj[idx,2] = 0
            pn[i,0] = board_n
    return pl,pr,pobj,pn

def __intrinsics(p,pobj,pn,img_size,params=__p_cam):
    n = pn.height
    intrinsics = cv.CreateMat(3,3,cv.CV_32FC1)
    distortion = cv.CreateMat(4,1,cv.CV_32FC1)
    rotations = cv.CreateMat(n,3,cv.CV_32FC1)
    translations = cv.CreateMat(n,3,cv.CV_32FC1)
    cv.Zero(rotations)
    cv.Zero(translations)
    cv.CalibrateCamera2(pobj,p,pn,img_size,intrinsics,distortion,rotations,translations,params)
    return intrinsics,distortion

def __extrinsics(pl,pr,pobj,pn,Cl,Cr,dl,dr,img_size,params=__p_rig):
    F = cv.CreateMat(3,3,cv.CV_32FC1)
    E = cv.CreateMat(3,3,cv.CV_32FC1)
    R = cv.CreateMat(3,3,cv.CV_64F)
    T = cv.CreateMat(3,1,cv.CV_64F)
    cv.Zero(F)
    cv.Zero(E)
    cv.Zero(R)
    cv.Zero(T)
    cv.StereoCalibrate(pobj,pl,pr,pn,Cl,dl,Cr,dr,img_size,R,T,E,F, \
                       (cv.CV_TERMCRIT_ITER+cv.CV_TERMCRIT_EPS,10000,1e-10),params)
    Pl = cv.CreateMat(3,4,cv.CV_64F)
    Pr = cv.CreateMat(3,4,cv.CV_64F)
    Rl = cv.CreateMat(3,3,cv.CV_64F)
    Rr = cv.CreateMat(3,3,cv.CV_64F)
    Q = cv.CreateMat(4,4,cv.CV_64F)
    cv.Zero(Pl)
    cv.Zero(Pr)
    cv.Zero(Rl)
    cv.Zero(Rr)
    cv.Zero(Q)
    cv.StereoRectify(Cl,Cr,dl,dr,img_size,R,T,Rl,Rr,Pl,Pr,Q,alpha=-1)
    return R,T,F,E,Pl,Pr,Rl,Rr,Q

def __persist(Cl,Cr,dl,dr,R,T,F,E,Pl,Pr,Rl,Rr,Q,outputdir):
    outputdir += "/"
    for path,mat in [("Cl.xml",Cl),("Cr.xml",Cr),("dl.xml",dl),("dr.xml",dr),("R.xml",R),("T.xml",T),("F.xml",F), \
                     ("E.xml",E),("Pl.xml",Pl),("Pr.xml",Pr),("Rl.xml",Rl),("Rr.xml",Rr),("Q.xml",Q)]:
        cv.Save(outputdir+path,mat)

def __error(pl,pr,Cl,Cr,dl,dr,F,Rl,Rr,Pl,Pr,n_board,board_size):
    board_n = np.prod(board_size)
    line_l = cv.CreateMat(board_n,3,cv.CV_32FC1)
    line_r = cv.CreateMat(board_n,3,cv.CV_32FC1)
    pl_new = cv.CreateMat(board_n,1,cv.CV_32FC2)
    pr_new = cv.CreateMat(board_n,1,cv.CV_32FC2)
    pl_reproj = cv.CreateMat(board_n,1,cv.CV_32FC2)
    pr_reproj = cv.CreateMat(board_n,1,cv.CV_32FC2)
    err_total = 0.0
    for i in range(n_board):
        for j in range(board_n):
            idx = i*board_n+j
            pl_new[j,0] = pl[idx,0]
            pr_new[j,0] = pr[idx,0]
        cv.UndistortPoints(pl_new,pl_reproj,Cl,dl,Rl,Pl)
        cv.UndistortPoints(pr_new,pr_reproj,Cr,dr,Rr,Pr)
        cv.ComputeCorrespondEpilines(pl_reproj,1,F,line_l)
        cv.ComputeCorrespondEpilines(pr_reproj,2,F,line_r)
        err = 0.0
        for j in range(board_n):
            err += (pl_reproj[j,0][0]*line_r[j,0]+pl_reproj[j,0][1]*line_r[j,1]+line_r[j,2])**2
            err += (pr_reproj[j,0][0]*line_l[j,0]+pr_reproj[j,0][1]*line_l[j,1]+line_l[j,2])**2
        err_total += err
    err_total = math.sqrt(err_total/board_n/2.0/n_board)
    return err_total

def load(directory):
    directory += "/"
    matrices = []
    for path in ["Cl.xml","Cr.xml","dl.xml","dr.xml","R.xml","T.xml","F.xml","E.xml","Pl.xml", \
                 "Pr.xml","Rl.xml","Rr.xml","Q.xml"]:
        matrices.append(cv.Load(directory+path))
    return matrices

def calibrate(imgpairs,board_size,outdir):
    img_size = cv.GetSize(imgpairs[0][0])
    corner_pairs = __detect(imgpairs,board_size)
    pl,pr,pobj,pn = __points(corner_pairs,board_size)
    Cl,dl = __intrinsics(pl,pobj,pn,img_size)
    Cr,dr = __intrinsics(pr,pobj,pn,img_size)
    R,T,F,E,Pl,Pr,Rl,Rr,Q = __extrinsics(pl,pr,pobj,pn,Cl,Cr,dl,dr,img_size)
    __persist(Cl,Cr,dl,dr,R,T,F,E,Pl,Pr,Rl,Rr,Q,outdir)
    n_board = len(corner_pairs)
    error = __error(pl,pr,Cl,Cr,dl,dr,F,Rl,Rr,Pl,Pr,n_board,board_size)
    return error
