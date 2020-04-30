import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from maskcontours import findmaskcontours, findcornerpoints
import random
from os import listdir
from os.path import isfile, join
from TrainingSiftPoints import sift_trainpoints
from IoU import compute_iou


def sift_detection2(select, trainsiftpoints, sift=1, draw=1, drawlines=1, matchtresh=10):
    # ---------------------------------------------------------------------------------------------------------------
    # This function includes the main algorithm that matches keypoints from the template images to the sample image
    # The function output is defined as IoU if something is found otherwise output is 0.
    # ---------------------------------------------------------------------------------------------------------------

    # tweakable
    MIN_MATCH_COUNT = matchtresh

    # computing time counter
    start = time.time()

    # read the sample image using opencv
    img2 = cv2.imread('./WashingtonOBRace/img_{}.png'.format(select))  # detect gate image

    # set imgmask resolution, THIS IS ASSUMED CONSTANT FOR THIS ASSIGNMENT!
    image_res = [360, 360]

    # find the contours of the mask of img2 if it exists
    nomask = False
    try:
        contoursmask = findmaskcontours(select)  # for the detect image (necessary for reliability calculation)
    except:
        nomask = True
        print(select)

    # Create either sift or surf object (based on the function input
    if sift:
        sift = cv2.xfeatures2d.SIFT_create()
    else:
        sift = cv2.xfeatures2d.SURF_create(400)

    # find the keypoints and descriptors with sift object for sample image
    kp2, des2 = sift.detectAndCompute(img2, None)

    # feature matching is done using FLANN
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # find a list of matches giving a list entry for every trainimage sift points collection
    matches = []
    for i in range(0, len(trainsiftpoints[0])):
        matches.append(flann.knnMatch(trainsiftpoints[3][i], des2, k=2))

    # Now each training image has its own set of matches and for each of them Lowe's ratio test is performed
    # if the amount of good points is better than for the previous training image it is saved otherwise discarded

    good = []
    for i in range(0, len(trainsiftpoints[0])):
        goodcur = []
        for m, n in matches[i]:
            if m.distance < 0.7 * n.distance:
                goodcur.append(m)
        if len(goodcur) > len(good):
            good = goodcur
            img1 = trainsiftpoints[0][i]
            kp1 = trainsiftpoints[2][i]
            cornlisttrain1 = trainsiftpoints[1][i]


    # next part is for visualization of the results

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = img1.shape
        pts = np.float32(cornlisttrain1).reshape(-1, 1, 2)

        # basic calculations done elapsed time:
        elapsed_time = (time.time() - start)

        print()
        print("algorithm time: {} seconds".format(elapsed_time))

        if nomask == True:  # we know that for this case -> no mask == no gate
            # since a gate has been found at this point this should always be a false positive
            output = 2
            return output

        try:
            dst = cv2.perspectiveTransform(pts, M)

            cv2.drawContours(img2, contoursmask, 0, (0, 0, 255), 2)
            img2 = cv2.polylines(img2, [np.int32(dst)], True, (0, 255, 0), 2, cv2.LINE_AA)
        except:
            error = True
            print()
            print(error)
    else:
        matchesMask = None
        output = 0
        print("no gate found")
        return output

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    if draw and drawlines:
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
        plt.imshow(img3)
        plt.show()
        try:
            iou_now = compute_iou(contoursmask, [dst[0][0], dst[1][0], dst[2][0], dst[3][0]])
            print("intersection over Union: {}".format(iou_now))
            print('image: {}'.format(select))
            output = iou_now
        except:
            print("couldnt compute iou")
            print('image: {}'.format(select))
            output = 2
    elif draw:
        plt.imshow(img2)
        # plt.ion()
        plt.show()
        plt.draw()
        plt.pause(0.001)
        try:
            iou_now = compute_iou(contoursmask, [dst[0][0], dst[1][0], dst[2][0], dst[3][0]])
            print("intersection over Union: {}".format(iou_now))
            print('image: {}'.format(select))
            output = iou_now
        except:
            print("couldnt compute iou")
            print('image: {}'.format(select))
            output = 2
        time.sleep(0.1)
    else:
        try:
            iou_now = compute_iou(contoursmask, [dst[0][0], dst[1][0], dst[2][0], dst[3][0]])
            print("intersection over Union: {}".format(iou_now))
            print('image: {}'.format(select))
            output = iou_now
        except:
            print("couldnt compute iou")
            print('image: {}'.format(select))
            output = 2
    return output


if __name__ == '__main__':
    # ---------------------------------------------------------------------------------------------------------------
    # Here is included a sample code of how the function sift_detection2 can be used

    dir = "C:\\Users\\Robbin\\PycharmProjects\\ObjectDetectionMAV\\WashingtonOBRace"
    files = [f.split('_')[1].split('.')[0] for f in listdir(dir) if isfile(join(dir, f))]

    trainlist = sift_trainpoints(['8', '13', '27', '48', '60'], 1)

    sift_detection2('11', trainlist, 1, 1, 1)