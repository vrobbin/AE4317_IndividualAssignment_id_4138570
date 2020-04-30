import cv2
import numpy as np
from maskcontours import findmaskcontours, findcornerpoints


def compute_iou(contour1, cornpoints):
    # --------------------------------------------------------------------------------------------------------------
    # contour1, contour2 represent the contourvectors for both the inputs contour1 and cornpoints respectively where
    # contour will be calculated as the combined contour
    # --------------------------------------------------------------------------------------------------------------

    # First define a new contour by drawing both contours on a black image and creating a new contour
    # this new contour will have Area = Area(contour1) + Area(contour2) - 2*Area(intersection)
    res = [500, 500]
    blank_image = np.zeros((res[0], res[1], 3), np.uint8)
    blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)

    # plt.imshow(blank_image, 'gray')
    # plt.show()

    # draw contour from cornerpoints
    blank_image = cv2.line(blank_image, (cornpoints[0][0], cornpoints[0][1]), (cornpoints[1][0], cornpoints[1][1]), 255, 1)
    blank_image = cv2.line(blank_image, (cornpoints[1][0], cornpoints[1][1]), (cornpoints[2][0], cornpoints[2][1]), 255, 1)
    blank_image = cv2.line(blank_image, (cornpoints[2][0], cornpoints[2][1]), (cornpoints[3][0], cornpoints[3][1]), 255, 1)
    blank_image = cv2.line(blank_image, (cornpoints[3][0], cornpoints[3][1]), (cornpoints[0][0], cornpoints[0][1]), 255, 1)

    # convert to contour

    gray_image = blank_image

    # Apply cv2.threshold() to get a binary image
    ret, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)

    # Find contours:
    im, contour2, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(blank_image, contour2, 0, (255), 1)

    cont2area = cv2.contourArea(contour2[0])

    # plt.imshow(blank_image, 'gray')
    # plt.show()

    # draw contour
    cv2.drawContours(blank_image, contour1, 0, (255), 1)

    # We save the file locally as a quick fix to get rid of the different layers at which has been drawn thus far
    cv2.imwrite('cont.png', blank_image)

    # plt.imshow(blank_image, 'gray')
    # plt.show()

    # then we reload the image to find the combined contour
    image = cv2.imread('cont.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply cv2.threshold() to get a binary image
    ret, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)

    # Find contours:
    im, contour, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    blank_image = cv2.cvtColor(blank_image, cv2.COLOR_GRAY2BGR)

    # draw total contours:
    cv2.drawContours(blank_image, contour, 0, (0, 255, 0), 1)

    # plt.imshow(blank_image, 'gray')
    # plt.show()

    # now all the areas can be determined and the IoU is given as (area1+area2-areatotal)/areatotal
    areatotal = cv2.contourArea(contour[0])
    areacont1 = cv2.contourArea(contour1[0])
    areacont2 = cv2.contourArea(contour2[0])

    iou = (areacont1+areacont2-areatotal)/areatotal

    return iou


if __name__ == "__main__":

    # Example usage of IoU function:
    cont1 = findmaskcontours('11')
    cont2 = findmaskcontours('12')
    cornerpoints = findcornerpoints(cont2, [360, 360])

    iou = compute_iou(cont1, cornerpoints)

    print(iou)