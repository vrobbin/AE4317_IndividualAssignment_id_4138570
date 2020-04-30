import cv2
import math


def findmaskcontours(imnum):
    # convert input to string
    imnums = str(imnum)

    # Load the image and convert it to grayscale:
    image = cv2.imread("./WashingtonOBRace/masksed/mask_{}.png".format(imnums))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply cv2.threshold() to get a binary image
    ret, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)

    # Find contours:
    im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return contours


def findcornerpoints(contours, imresxy):
    # find the cornerpoints of a contour array
    # done by assuming that points closest to corners of square image itself are cornerpoints
    array = contours[0]
    cornlist = [[0, 0], [0, 0], [0, 0], [0, 0]]  # initialize cornerpoints xy values
    corndistlist = [10000, 10000, 10000, 10000]  # initialize distance to image corner large

    for i in range(0, len(array)):
        lefttop = math.sqrt((array[i][0][0]) ** 2 + (array[i][0][1]) ** 2)
        leftbot = math.sqrt((array[i][0][0]) ** 2 + (imresxy[1] - (array[i][0][1])) ** 2)
        rightbot = math.sqrt((imresxy[0] - (array[i][0][0])) ** 2 + (imresxy[1] - (array[i][0][1])) ** 2)
        righttop = math.sqrt((imresxy[0] - (array[i][0][0])) ** 2 + (array[i][0][1]) ** 2)
        if lefttop < corndistlist[0]:
            corndistlist[0] = lefttop
            cornlist[0] = [array[i][0][0], array[i][0][1]]
        elif leftbot < corndistlist[1]:
            corndistlist[1] = leftbot
            cornlist[1] = [array[i][0][0], array[i][0][1]]
        elif rightbot < corndistlist[2]:
            corndistlist[2] = rightbot
            cornlist[2] = [array[i][0][0], array[i][0][1]]
        elif righttop < corndistlist[3]:
            corndistlist[3] = righttop
            cornlist[3] = [array[i][0][0], array[i][0][1]]
    return cornlist


if __name__ == "__main__":
    print("unk")
