########################################################################################################################
# OS walks through absolute paths of image data and stores file location of JPEG with label name and label encoding.   #
# Also has a function definition to extract 9 unique, random images and plot them on one figure object.                #
########################################################################################################################

import os.path, os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from PIL import Image


# Constants
stages = ['Battlefield', 'Final Destination', 'Kalos Pokemon League', 'Lylat Cruise', 'Pokemon Stadium 2',
          'Smashville', 'Unova Pokemon League', 'Yoshis Island', 'Yoshis Story']


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


def plot_images(df):
    index_images = []
    for i in range(9):
        index_images.append(df.index[df['encoded labels'] == i].tolist()[0])

    images_to_plot = 9
    sample_images = df.iloc[index_images, 0]
    sample_labels = df.iloc[index_images, 1]

    plt.style.use('seaborn-muted')

    fig, axes = plt.subplots(3, 3,
                             figsize=(25, 25),
                             sharex=True, sharey=True)  # https://stackoverflow.com/q/44703433/1870832

    for i in range(images_to_plot):
        # axes (subplot) objects are stored in 2d array, accessed with axes[row,col]
        subplot_row = i // 3
        subplot_col = i % 3
        ax = axes[subplot_row, subplot_col]

        # plot image on subplot
        im = Image.open(sample_images.iloc[i])
        ax.imshow(im)

        ax.set_title('Stage Name: {}'.format(sample_labels.iloc[i]))
        # ax.set_xbound([0, 28])

    plt.tight_layout()
    plt.show()


# filename = "C:/Users/17073/Desktop/SSBU/Labels/"
filename = os.getcwd() + "/Labels/"
file_list = []
stage_list = []

extract_JPG(filename)

# encode labels
labelencoder = LabelEncoder()
encoded_label = ((labelencoder.fit_transform(stage_list)).tolist())

# create dataframe
df = pd.DataFrame(zip(file_list, stage_list, encoded_label), columns=['file paths', 'stage types', 'encoded labels'])
# save to csv
df.to_csv('SSBU_data.csv', index=False)

# plot_images(df)
