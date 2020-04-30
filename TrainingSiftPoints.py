import cv2
from maskcontours import findmaskcontours, findcornerpoints


def sift_trainpoints(masks, sift):
    # -----------------------------------------------------------------------------------------------------------------
    # This Function finds list the keypoints and descriptor for all images in "masks" (which is a list of strings with
    # numbers for each of the template images, sift can be set to True or False for using either SIFT or SURF
    # -----------------------------------------------------------------------------------------------------------------

    trainList = []

    # for the masks we read all the training images into list of images maskimglist
    maskimglist = []
    for im in masks:
        maskimglist.append(cv2.imread('./WashingtonOBRace/img_{}.png'.format(im), 0))  # read images

    # set imgmask resolution, THIS IS ASSUMED CONSTANT FOR THIS ASSIGNMENT!
    image_res = [360, 360]

    # find the list of contours for all the training images
    contoursmasktrain = []
    cornlisttrain = []
    ind = 0
    for im in masks:
        contoursmasktrain.append(findmaskcontours(im))
        cornlisttrain.append(findcornerpoints(contoursmasktrain[ind], image_res))
        ind += 1

    # read all masks for training images
    maskstrain = []
    for im in masks:
        maskstrain.append(cv2.imread('./WashingtonOBRace/masksed/mask_{}.png'.format(im), 0))

    # Create either sift or surf object (based on the function input
    if sift:
        sift = cv2.xfeatures2d.SIFT_create()
    else:
        sift = cv2.xfeatures2d.SURF_create(400)

    # Find list of keypoints and descripters for training images with sift object
    kp = []
    des = []
    for i in range(0, len(masks)):
        kpcur, descur = sift.detectAndCompute(maskimglist[i], maskstrain[i])
        kp.append(kpcur)
        des.append(descur)

    # create outputList
    trainList = [maskimglist, cornlisttrain, kp, des]

    # return trainlist
    return trainList


if __name__ == "__main__":
    print("Unk")
