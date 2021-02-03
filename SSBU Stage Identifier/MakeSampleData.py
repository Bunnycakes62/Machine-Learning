########################################################################################################################
# Makes Sample Data. Just what it sounds like                                                                          #
########################################################################################################################
import cv2
import os.path, os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Constants
width = 512
height = 288
dim = (width, height)
stages = ['Battlefield', 'Final Destination', 'Kalos Pokemon League', 'Lylat Cruise', 'Pokemon Stadium 2',
          'Smashville', 'Unova Pokemon League', 'Yoshis Island', 'Yoshis Story']

# Generate Images From Video File
#
# filename = "C:/Users/17073/Desktop/SSBU/Videos/Glitch 7/Glitch 7 SSBU - Xerom  (Ness) Vs. JeBB (Palutena) Smash Ultimate Tournament Pools-_ZyVgBufnL0.mp4"
#
# vidcap = cv2.VideoCapture(filename)
# success, image = vidcap.read()
#
# # Set up parameters
# seconds = 20  # 5 seconds for 1v1, 20 seconds for tourneys
# fps = vidcap.get(cv2.CAP_PROP_FPS)  # Gets the frames per second
# multiplier = int(round(fps * seconds))
# dataPath = "C:/Users/17073/Desktop/SSBU/SampleData/"
# # print("Setting Up Parameters...")
#
# # Initiate process
# while success:
#     # current frame number rounded because frame intervals can take non-integer values
#     frameId = int(round(vidcap.get(1)))
#     success, image = vidcap.read()
#
#     if frameId % multiplier == 0:
#         resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
#         gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#         cv2.imwrite(os.path.join(dataPath, "frame%d.jpg" % (frameId)), gray)
#         print("Adding Images to Data Path...", frameId)
#
# vidcap.release()
# print("Complete")

# Store to Dataframe

def extract_JPG(filename):
    # extract only .jpg to file_list
    for root, dirs, files in os.walk(filename):
        for file in files:
            if file.endswith(".jpg"):
                name = format_string(root) + "/" + file
                file_list.append(name)
                label_maker(file_list[-1])


def label_maker(filename):
    if any(stage in filename for stage in stages):
        ind = list(map(lambda x: x == True, [stage in filename for stage in stages])).index(True)
        print(stages[ind])
        stage_list.append(stages[ind])


def format_string(value):
    value = value.replace('\\', '/')
    return value


filename = os.getcwd() + "/SampleData/"
file_list = []
stage_list = []

extract_JPG(filename)

# encode labels
# labelencoder = LabelEncoder()
# encoded_label = ((labelencoder.fit_transform(stage_list)).tolist())
encoded_label = []
# create dataframe
df = pd.DataFrame(zip(file_list, stage_list), columns=['file paths', 'stage types'])

# make a subset for toy model
for i in range(df.shape[0]):
    if 'Battlefield' == df.iloc[i,1]:
        encoded_label.append(0)
    elif 'Kalos Pokemon League' == df.iloc[i,1]:
        encoded_label.append(2)
    elif 'Pokemon Stadium 2' == df.iloc[i, 1]:
        encoded_label.append(4)

df['encoded labels'] = encoded_label
# save to csv
df.to_csv('sample_data.csv', index=False)
