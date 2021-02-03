# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from numpy.random import seed
from tensorflow import set_random_seed

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense
from keras.models import Model
from sklearn.metrics import confusion_matrix

# Turn off GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

seed(1)
set_random_seed(1)

artists = pd.read_csv('../artists.csv')

# Preprocess Data
# Sort artists by number of paintings
artists = artists.sort_values(by=['paintings'], ascending=False)

# Create a dataframe with artists having more than 200 paintings
artists_top = artists[artists['paintings'] >= 200].reset_index()
artists_top = artists_top[['name', 'paintings']]
artists_top['class_weight'] = artists_top.paintings.sum() / (artists_top.shape[0] * artists_top.paintings)
class_weights = artists_top['class_weight'].to_dict()
# Struggling to identify name
updated_name = "Albrecht_Dürer".replace("_", " ")
artists_top.iloc[4, 0] = updated_name

# Explore images of top artists
images_dir = '../images/images'
artists_dirs = os.listdir(images_dir)
artists_top_name = artists_top['name'].str.replace(' ', '_').values

# See if all directories exist
for name in artists_top_name:
    if os.path.exists(os.path.join(images_dir, name)):
        print("Found -->", os.path.join(images_dir, name))
    else:
        print("Did not find -->", os.path.join(images_dir, name))

# Augment data
batch_size = 8
train_input_shape = (224, 224, 3)
n_classes = artists_top.shape[0]

train_datagen = ImageDataGenerator(validation_split=0.2,
                                   rescale=1./255.,
                                   shear_range=5,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                  )

train_generator = train_datagen.flow_from_directory(directory=images_dir,
                                                    class_mode='categorical',
                                                    target_size=train_input_shape[0:2],
                                                    batch_size=batch_size,
                                                    subset="training",
                                                    shuffle=True,
                                                    classes=artists_top_name.tolist()
                                                   )

valid_generator = train_datagen.flow_from_directory(directory=images_dir,
                                                    class_mode='categorical',
                                                    target_size=train_input_shape[0:2],
                                                    batch_size=batch_size,
                                                    subset="validation",
                                                    shuffle=True,
                                                    classes=artists_top_name.tolist()
                                                   )

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
print("Total number of batches =", STEP_SIZE_TRAIN, "and", STEP_SIZE_VALID)

# Print a random paintings and it's random augmented version
fig, axes = plt.subplots(1, 2, figsize=(20,10))

random_artist = random.choice(artists_top_name)
random_image = random.choice(os.listdir(os.path.join(images_dir, random_artist)))
random_image_file = os.path.join(images_dir, random_artist, random_image)

# Original image
image = plt.imread(random_image_file)
axes[0].imshow(image)
axes[0].set_title("An original Image of " + random_artist.replace('_', ' '))
axes[0].axis('off')

# Transformed image
aug_image = train_datagen.random_transform(image)
axes[1].imshow(aug_image)
axes[1].set_title("A transformed Image of " + random_artist.replace('_', ' '))
axes[1].axis('off')

plt.show()

# Build the Autoencoder

# target_size = (224,224,3)
# x = Lambda(lambda image: image.resize(image, target_size))(input)
input_img = Input(shape=(224, 224, 3))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (28, 28, 4) i.e. 3136-dimensional

x = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
decoded = Dense(11, activation='sigmoid')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

autoencoder.fit(train_generator,
                epochs=5,
                # batch_size=16,
                shuffle=True,
                validation_data=valid_generator)

plt.plot(autoencoder.history.history['loss'])
plt.plot(autoencoder.history.history['accuracy'])
plt.show()

score = autoencoder.evaluate_generator(valid_generator, verbose=1)
print("Prediction accuracy on validation data =", score[1])

predict = autoencoder.predict(valid_generator)
predicted_class_indices = np.argmax(predict, axis=1)
labels = [valid_generator.classes[k] for k in valid_generator.index_array]
# labels2 = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
print(confusion_matrix(predicted_class_indices, labels))

# url = 'https://upload.wikimedia.org/wikipedia/commons/8/81/Edgar_Degas_-_The_Ballet_Class_-_Google_Art_Project.jpg'
#
# import imageio
# import cv2
#
# web_image = imageio.imread(url)
# web_image = cv2.resize(web_image, dsize=train_input_shape[0:2], )
# web_image = image.img_to_array(web_image)
# web_image /= 255.
# web_image = np.expand_dims(web_image, axis=0)
#
#
# prediction = autoencoder.predict(web_image)
# prediction_probability = np.amax(prediction)
# prediction_idx = np.argmax(prediction)
#
# print("Predicted artist =", labels[prediction_idx].replace('_', ' '))
# print("Prediction probability =", prediction_probability*100, "%")
#
# plt.imshow(imageio.imread(url))
# plt.axis('off')
# plt.show()

autoencoder.save('autoencoderArtist.h5')
del autoencoder
# model = keras.models.load_model('encodedArtist.h5')
