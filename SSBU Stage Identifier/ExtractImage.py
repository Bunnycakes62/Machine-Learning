########################################################################################################################
# os walks through video folders, creating directory list of video filenames. It then extracts an image every n seconds#
# from the video, resizes it to 512x288, and grayscales before saving the image to the relevant image directory.       #
########################################################################################################################

import cv2
import os

# Constants
width = 512
height = 288
dim = (width, height)

# Recursively grab all files
Stages = ['Battlefield', 'Final Destination', 'Kalos Pokemon League', 'Lylat Cruise', 'Pokemon Stadium 2',
          'Smashville', 'Unova Pokemon League', 'Yoshis Island', 'Yoshis Story']

dirList = []

filename = "C:/Users/17073/Desktop/SSBU/Labels/"

for st in Stages:
    # dirList.append("C:/Users/17073/Desktop/SSBU/Labels/" + st + "/Videos/*")
    for root, dirs, files in os.walk(filename + st + "/Videos/"):
        for file in files:
            dirList.append(filename + st + "/Videos/" + file)

for st in Stages:
    print("Adding images for", st)
    for d in dirList:
        print("Generating Directory List...")
        if st in d:
            videoFile = d
            vidcap = cv2.VideoCapture(videoFile)
            success, image = vidcap.read()

            # Set up parameters
            seconds = 5  # 5 seconds for 1v1, 20 seconds for tourneys
            fps = vidcap.get(cv2.CAP_PROP_FPS)  # Gets the frames per second
            multiplier = int(round(fps * seconds))
            dataPath = "C:/Users/17073/Desktop/SSBU/Labels/" + st + "/Images/"
            # print("Setting Up Parameters...")

            # Initiate process
            while success:
                # current frame number rounded because frame intervals can take non-integer values
                frameId = int(round(vidcap.get(1)))
                success, image = vidcap.read()

                if frameId % multiplier == 0:
                    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
                    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(os.path.join(dataPath, "frame%d.jpg" % (frameId + dirList.index(d))), gray)
                    print("Adding Images to Data Path...", frameId)

            vidcap.release()
        print("Complete")

# modified from the following links
# file path: https://stackoverflow.com/questions/41586429/opencv-saving-images-to-a-particular-folder-of-choice/41587740
# sampling: https://stackoverflow.com/questions/22704936/reading-every-nth-frame-from-videocapture-in-opencv

