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

# Add paired images vertically
combined = addImage(deskIms[0], deskIms[3], finalSize)
combined2 = addImage(deskIms[1], deskIms[2], finalSize)

# Horizontally concat the tall images together.
combined3 = addImage(combined, combined2.astype("uint8"), finalSize)

# Remove the extra black background
fin = cropOutBlack(combined3.astype("uint8"))

cv2.imwrite("keyIm.jpg", fin)

# drawKeypointsAcross(cropOutBlack(combined), cropOutBlack(combined2))