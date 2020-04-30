import matplotlib.pyplot as plt
from SIFTDetectionNew import sift_detection2
from CollectTrainingData import collecttrainingdata
import os
from TrainingSiftPoints import sift_trainpoints

# This file is used to generate ROC curves

#---------------- SET VARIABLES ----------------------------------------------------------------------------

# IoU treshold
iou_tresh = 0.7

# Selected TEMPLATE images
templates = ['8', '13', '27', '48', '60', '73', '113', '126', '131', '141', '165', '198', '243', '239', '249', '262', '276', '324', '375', '426', '388']

#-----------------------------------------------------------------------------------------------------------

# set current directory to current file directory so we start in the right folder
os.chdir(os.path.dirname(__file__))
print(__file__)

# First the trainingdata as well as the images for detection are collected
data = collecttrainingdata("./WashingtonOBRace", templates)

# find the siftpoints for the template list once

trainlist = sift_trainpoints(data[0], 1)

# # create a dictionary where the keys will be the images names and the values will be a string saying
# # 'tp' true positive, 'tn' true negative, 'fp' false positive, 'fn' false negative
#
# rocdict = dict.fromkeys(data[1] , 0) # note 0 is not an actual value (tp, fn, etc..) and just used for initialization

# we also need a list of images that are considered negative (no gate on the image)
# these were created manually and a list of them is given by:
negList = ['15', '49', '75', '118', '206', '245', '377', '381', '500', '501', '600', '601', '602', '603', '604', '605',
           '606', '607', '608', '609', '700', '701', '702', '703', '704', '705', '706', '707', '708', '709', '800',
           '801', '802', '803', '804', '805', '806', '807', '808', '809', '900', '901', '902', '903', '904', '905',
           '907', '908', '909']

# furthermore the IoU will determine if the detection is a true positive or a false positive
# if the siftdetection detects it hasnt found a gate and that means false negative if the image is not in the negList
# if the image is in the negList it is considered a true negative

# determine the values of the dictionary:

# Now run the sift detection method for all files in data[1] (contains files where we want to find gates or not)

# do this in a loop for using different tresholds to determine roc curve

TPR = []
FPR = []
tresholds = [33, 25, 7, 3, 1]

for thresh in tresholds:

    # counting
    tp_count = 0
    fp_count = 0
    tn_count = 0
    fn_count = 0
    error_count = 0

    for i in data[1]:
        iou_cur = (sift_detection2(i, trainlist, 1, 0, 0, thresh))
        if iou_cur == 2:  # we are dealing with a false positive since we found something but no IoU could be computed
            fp_count += 1
        elif iou_cur > 0:  # we know we are dealing with a true positive or false positive based on the IoU
            if iou_cur > iou_tresh:
                tp_count += 1
            else:
                fp_count += 1
        elif iou_cur < 0:  # this shouldn't happen so lets keep track of it
            error_count += 1
        else:  # we know we are dealing with a false negative or a true negative
            for ims in negList:  # if current image is in the negList this is a true negative
                if i == ims:
                    tn_count += 1
                    break
            fn_count += 1  # else it is a false negative

    print()
    print("tp: " + str(tp_count))
    print()
    print("fp: " + str(fp_count))
    print()
    print("tn: " + str(tn_count))
    print()
    print("fn: " + str(fn_count))
    print()
    print("error: " + str(error_count))

    # sensitivity and specificity

    TPR.append(tp_count / (tp_count + fn_count))
    FPR.append(fp_count / (fp_count + tn_count))

    print()
    print(TPR)
    print()
    print(FPR)

# plot the ROC curve
print()
print(TPR)
print()
print(FPR)

TPR.insert(0, 0)
FPR.insert(0, 0)

plt.plot(FPR, TPR)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("False Positive Ratio")
plt.ylabel("True Positive Ratio")
plt.show()
