# Written by Andy, copied almost entirely from Aiden Rosebrock the bald CV python guy:
# https://www.pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/
import cv2
import numpy as np
import threading

# Get webcam stream                         DONE 3:20
# Get other webcam stream                   FAILED, BAD WEBCAM
# write threaded code for loading the webcams      DONE 3:45
# get stitcher working for 2 ims
# get stither working for 2 streams.


# # Start webcam captures
# NUM_CAMERAS = 2
# CAM_BUFFERS = [0]*NUM_CAMERAS
#
# def threadedRetrieveCamFrame(id):
#     print("Threaded webcam started with id: ", id)
#     cap = cv2.VideoCapture(id)
#     while True:
#         ret, frame = cap.read()
#         CAM_BUFFERS[id] = frame
#
# for i in range(NUM_CAMERAS):
#     t = threading.Thread(target=threadedRetrieveCamFrame, args=([i]), daemon=True)
#     t.start()


# Single image stitching test
# rightIm = cv2.imread("ims/right3.JPG")
# leftIm = cv2.imread("ims/left3.JPG")
# rightIm = cv2.imread("ims/right.JPG")
# leftIm = cv2.imread("ims/left.JPG")
im1 = cv2.imread("ims/3ims1.JPG")
im2 = cv2.imread("ims/3ims2.JPG")
im3 = cv2.imread("ims/3ims3.JPG")

def showIm(im):
    cv2.imshow("im", im)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()


# stitcher = cv2.Stitcher.create()
#
# foo = cv2.imread("ims/foo.png")
# bar = cv2.imread("ims/bar.png")
# (status, stitched) = stitcher.stitch((foo,bar))
#
#
# # (status, stitched) = cv2.Stitcher.stitch((foo, bar))
#
# print("status:", status)


# stitcher = cv2.Stitcher_create()
foo = cv2.imread("ims/foo.png")
bar = cv2.imread("ims/bar.png")


# stitcher = cv2.Stitcher.create(cv2.S titcher_PANORAMA)
stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
# stitcher.setRegistrationResol(0.3)
stitcher.setPanoConfidenceThresh(0.3) # might be too aggressive for real examples
# stitcher.setSeamEstimationResol(0.01)
stitcher

status, result = stitcher.stitch((foo,bar))
# status, result = stitcher.stitch((rightIm, leftIm))
# status, result = stitcher.stitch((im1, im2, im3))
print(status)

# assert status == 0 # Verify returned status is 'success'
# cv2.imshow("result", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# if the status is '0', then OpenCV successfully performed image stitching
if status == 0:
    # write the output stitched image to disk
    cv2.imwrite("outIm.jpg", result)
    # display the output stitched image to our screen
    cv2.imshow("Stitched", result)
    cv2.waitKey(0)
# otherwise the stitching failed, likely due to not enough keypoints)
# being detected
else:
    print("[INFO] image stitching failed ({})".format(status))
