import cv2
import numpy as np


# About feature detection and matching:
    # kps is keypoints, it's literally a list of KeyPoints with .pt location, .angle, .octave, .size and some stuff
    # descs is descriptors of each keypoint. len(desc[i]) == len(kps[i]). Apparently when the descriptors are similar for different keypoints, it means that the keypoints are similar.

########### debug functions
# # Draw keypoints onto an image
# yea = cv2.drawKeypoints(deskIms[0], kps[0], deskIms[0].copy(), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imwrite("keyIm.jpg", yea)


# # For debugging keypoint matching: draw lines between two images to link matching descriptors
# img3 = cv2.drawMatches(deskIms[0],kps[0],deskIms[1],kps[1],matches01,None)
# cv2.imwrite("keyIm.jpg", img3)

# Init a FLANN feature matcher
FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5) # use for SIFT features
index_params = dict(algorithm = 6, # Use for ORB features
                    table_number = 6,
                    key_size = 12)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

def drawKeypointsAcross(im1, im2):
    orb = cv2.ORB_create()
    kp1, desc1 = orb.detectAndCompute(im1, None)
    kp2, desc2 = orb.detectAndCompute(im2, None)

    matches = flann.knnMatch(desc1, desc2, k=2)
    # matches = [x for x in matches if len(x) == 2]

    good_matches, _, _ = getGoodMatches(im1, im2)
    img3 = cv2.drawMatches(im1, kp1, im2, kp2, np.array(good_matches),None)
    cv2.imwrite("keyIm.jpg", img3)

# Using flann and the ratio test, return the best matches between two sets of descriptors
def find_good_matches(desc1, desc2):
    # Apply the flann
    matches = flann.knnMatch(desc1, desc2, k=2)
    matches = [x for x in matches if len(x) == 2]

    # Not all matches are good, filter out the bad uing lowe's ratio criterion from:
    # https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
    ratio_thresh = 0.75
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    return good_matches

# Calcuate descriptors for two images, and pull out the keypoints where they overlap
def getGoodMatches(im1, im2):
    # Create SIFT feature detector and use it to find keypoints in all images # Too slow!
    # sifter = cv2.SIFT_create()
    # kp1, desc1 = sifter.detectAndCompute(im1, None) # desc1.shape => (2934, 128)
    # kp2, desc2 = sifter.detectAndCompute(im2, None)

    orb = cv2.ORB_create()
    kp1, desc1 = orb.detectAndCompute(im1, None)
    kp2, desc2 = orb.detectAndCompute(im2, None)

    # Filter the descriptor matches to only get good ones
    # good_matches = find_good_matches(desc1.astype(np.float32), desc2.astype(np.float32))
    good_matches = find_good_matches(desc1, desc2)

    # Pull out the actual points that match
    kp1_points = [kp1[gmatch.queryIdx].pt for gmatch in good_matches]
    kp2_points = [kp2[gmatch.trainIdx].pt for gmatch in good_matches]
    return good_matches, kp1_points, kp2_points

def addImage(baseIm, newIm, finalSize):
    matches, kps_1, kps_2 = getGoodMatches(newIm, baseIm)
    homo, _ = cv2.findHomography(np.array(kps_1), np.array(kps_2),
                                 method=cv2.RANSAC, ransacReprojThreshold=2)

    h1, w1, _ = newIm.shape
    h2, w2, _ = baseIm.shape

    warped_image = cv2.warpPerspective(newIm, homo, finalSize)

    # print("baseim shape:", baseIm.shape)
    # print("finalsize:", finalSize)

    if (baseIm.shape[0] == finalSize[1] and baseIm.shape[1] == finalSize[0]):
        print("adding base image")
        # warped_image += baseIm*(warped_image == 0)
        baseIm += warped_image * (baseIm == 0)
        warped_image = baseIm
        # warped_image += baseIm*(warped_image == 0)
    else:
        warped_image[0:baseIm.shape[0], 0:baseIm.shape[1]] = baseIm
    cv2.imwrite("keyIm.jpg", warped_image)
    return warped_image


def cropOutBlack(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    return im[x:(x+w), y:(y+h)]


def perspectiveWarpPreserveBounds(im1, im2):
    # Find good descriptor matches between the two images
    matches, im1_keypoints, im2_keypoints = getGoodMatches(im1, im2)

    # Calculate the homography    source             destination
    homo, _ = cv2.findHomography(np.array(im1_keypoints), np.array(im2_keypoints),
                                 method=cv2.RANSAC, ransacReprojThreshold=2)

    # Find the corner bounds
    h, w, _ = im2.shape
    corners = np.array([[0, 0, 1],
                       [w, 0, 1],
                       [0, h, 1],
                       [w, h, 1]])
    x_corners = [np.dot(homo[0], corn) / np.dot(homo[2], corn) for corn in
                 corners]
    y_corners = [np.dot(homo[1], corn) / np.dot(homo[2], corn) for corn in
                 corners]
    scaling = [np.dot(homo[2], corn) for corn in corners]

    # change translation terms to make sure entire image falls into the box
    x_adj = min(x_corners) * scaling[x_corners.index(min(x_corners))]
    y_adj = min(y_corners) * scaling[y_corners.index(min(y_corners))]
    homo[0][-1] -= x_adj
    homo[1][-1] -= y_adj

    # Find new max x and y for the output image
    x_corners = [np.dot(homo[0], corn) / np.dot(homo[2], corn) for corn in
                 corners]
    y_corners = [np.dot(homo[1], corn) / np.dot(homo[2], corn) for corn in
                 corners]

    print("x_adj:", x_adj)
    print("y_adj:", y_adj)
    return cv2.warpPerspective(im2, homo, (int(max(x_corners)), int(max(y_corners))))
