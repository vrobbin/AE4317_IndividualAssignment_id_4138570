from os import listdir
from os.path import isfile, join


def collecttrainingdata(dir, trainlist):
    # -----------------------------------------------------------------------------------------------------------------
    # This function will take a directory and template images list as input and output the files in the directory that
    # are not already classified as template images together with the trainlist in an output list.
    # -----------------------------------------------------------------------------------------------------------------

    if not dir:
        dir = "./WashingtonOBRace"

    files = [f.split('_')[1].split('.')[0] for f in listdir(dir) if isfile(join(dir, f))]

    # determine the files list and remove the files that are already in the trainlist
    # using dictionary...

    dict1 = dict.fromkeys(files, 1)

    # delete template images:
    for i in trainlist:
        del dict1[i]

    # get a list from the keys of the dictionary, giving the list of images excluding the ones already assigned as
    # template images
    files_new = list(dict1.keys())

    output = [trainlist, files_new]

    return output


if __name__ == '__main__':
    collecttrainingdata([], ['8', '13', '27', '48', '60', '73', '113', '126', '131', '141', '165', '198', '243', '239',
                             '249', '262', '276', '324', '375', '426', '388'])
