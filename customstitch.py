# Tutorial from https://kushalvyas.github.io/stitching.html
import cv2
import numpy as np
from utils import *

# load in images
# deskIms = [cv2.imread("ims/" + file) for file in ["desk1.JPG", "desk2.JPG", "desk3.JPG", "desk4.JPG"]]
# deskIms = [cv2.imread("ims/desk3" + file) for file in ["1.JPG", "2.JPG", "3.JPG", "4.JPG"]]
deskIms = [cv2.imread("ims/desk4" + file) for file in ["1.JPG", "2.JPG", "3.JPG", "4.JPG"]]


h,w,_ = deskIms[0].shape
finalSize = (2*h,2*w)
combined = addImage(deskIms[0], deskIms[1], finalSize)
combined2 = addImage(combined, deskIms[3], finalSize)
combined3 = addImage(combined2, deskIms[2], finalSize)

fin = cropOutBlack(combined3)

cv2.imwrite("keyIm.jpg", fin)
