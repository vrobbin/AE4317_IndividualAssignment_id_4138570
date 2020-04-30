# Gate Detection for the Purpose of the AIRR Event

## Installation

To be able to run the code follow these steps:

> 1. Clone or download this git repository to your local machine and cd into the main folder
> 2. Within this main folder download and unzip *WashingtonOBRace.zip*, the main folder should now have all the *.py* files and a folder called *WashingtonOBRace* with inside it subfolders *masksed* and *masksorig*
> 2. Create and activate a virtual environment with Python 3.7 (earlier Python3 releases might also be ok)
> 3. Install dependencies into the virtual environment using *pip install -r requirements.txt*
> 4. cd into the main folder (the folder with all *.py* files) before running any scripts/functions.

## Generating ROC Curves:

To generate an ROC curve, the python script *RocCurves.py* should be used. Note generating an ROC curve may take a few minutes since it loops over all the images multiple times.

The main parameters that can be adjusted for this script are found at the top:

```python
# IoU treshold
iou_tresh = 0.7

# Selected TEMPLATE images
templates = ['8', '13', '27', '43', '60', '73', '113', '126', '131', '141', '165', '198', '239', '243', '249', '262', '276', '324', '375', '426', '388']
```

Increasing the amount of template images (defined in trainlist) and choosing them in a right way will significantly increase accuracy and result in more optimal ROC curves.

## Gate Detection on a Single Sample Image:

Running a detection on a single image can be done using the *sift_detection2* function found inside *SIFTDetectionNew.py*:

```python
def sift_detection2(select, trainsiftpoints, sift=1, draw=1, drawlines=1, matchtresh=10):
```

With the following input parameters:

> + **select** - An image number (given as a string)
> + **trainsiftpoints** - A list containing the keypoints and descriptors of a set of template images.
> 	* This list is obtained by running the list of template images through *sift_trainpoints( [list_of_template_images] )*
> + **sift** - *default=True* create SIFT object, *False* for SURF object (SURF is not properly tested/implemented yet)
> + **draw** - *default=True*, draws the detected gate with the contours of the sample image mask in blue and the actual detection in green
> + **drawlines** - *default=True*, draws lines between the template image and the sample image for all the matched keypoints
> + **matchtresh** - *default=10*, this is the threshold for what amount of matches a gate is considered detected

An example input is included at the bottom of the *SIFTDetectionNew.py* file which will be performed when running the code as a script.
