#!/usr/bin/python
import sys
import os
import mimetypes
import calib
import cv

def usage():
    print "Decapod - Camera Calibration"
    print sys.argv[0],"<dir_in> <board_w> <board_h> <dir_out>"
    print
    print "Performs stereo calibration using chessboard patterns of predefined size."
    print "The input directory must only contain the relevant images and corresponding images " + \
          "must appear next to each other in alphabetic order, i.e. left right left right..."
    print "  dir_in:   directory to read chessboard images from"
    print "  board_w:  number of chessboard squares on calibration pattern along width"
    print "  board_h:  number of chessboard squares on calibration pattern along height"
    print "  dir_out:  directory to persist calibration data to"
    sys.exit(1)

def calibrate(directory_in,board_size,directory_out):
    def imagepairs(directory):
        files_raw = os.listdir(directory)
        files_raw.sort()
        files = []
        for f in files_raw:
            type = mimetypes.guess_type(directory+"/"+f)[0]
            if str(type).startswith("image"):
                files.append(f)
        pairs = [(directory+"/"+files[i],directory+"/"+files[i+1]) for i in range(0,len(files),2)]
        pairs = map(lambda x: (cv.LoadImage(x[0]),cv.LoadImage(x[1])),pairs)
        return pairs
    d = os.path.dirname(directory_out)
    if not os.path.exists(d):
        os.makedirs(directory_out)
    print "Starting camera calibration..."
    print "  - specified board size as %ix%i" % board_size
    print "  - reading calibration images from",directory_in
    pairs = imagepairs(directory_in)
    print "  - detected",len(pairs),"stereo image pairs"
    print "  - performing camera calibration... (this may take several minutes)"
    error = calib.calibrate(pairs,board_size,directory_out)
    print "    ...done!"
    print "  - RMS reprojection error is",error,"pixels"
    print "  - output calibration data to",directory_out
    print "...finished!"

if len(sys.argv)!=5:
    usage()
directory_in,board_w,board_h,directory_out = sys.argv[1:]
board_size = int(board_w),int(board_h)
calibrate(directory_in,board_size,directory_out)
