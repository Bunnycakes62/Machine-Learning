# Task 1 (Creating a popart and compressible version of a colour image):
# Using either imageio or pillow to upload an image. read a colour image of your choosing or this default image (Links to an external site.).
# Please submit the image that you chose if not the default. Code could be:
# from imageio import imread
# im = imread(filename)
# from PIL import Image
# import numpy as np
# image = Image.open(filename)
# im = np.array(image)
# Each x and y pixel in the image should have 3 values corresponding to red green and blue values.
# Stack the pixels to create a dataset of shape (width*height, 3)
# Applying Kmeans to this dataset with k = 5, assigns each pixel to one of 5 centroids.
# Each of the five centroids corresponds to a colour.
# Plot the grayscale image of the indices of the pixel to centroid.
# Plot the colour image of the pixel mapped to their centroid.
# Plot the same images with k=10.

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

filename = 'supermarket.jpg'

image = Image.open(filename)
im = np.array(image)
reshaped = im.reshape(im.shape[0]*im.shape[1], 3)
plt.imshow(im)
plt.show()

# kmeans cluster for k = 5
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)

# Plot clusters to pixel data
grayImage = np.array(Image.open(filename).convert('LA'))
labels = kmeans.fit(reshaped).labels_
popImage = kmeans.cluster_centers_[labels].astype(int)

plt.scatter(grayImage[:, 0], grayImage[:, 1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1])
plt.title('k = 5')
plt.show()

# show color-mapped image
mapped = popImage.reshape(im.shape[0], im.shape[1], 3)
plt.imshow(mapped)
plt.show()

# kmeans cluster for k = 10
kmeans10 = KMeans(n_clusters=10, init='k-means++', max_iter=300, n_init=10, random_state=0)
# Plot clusters to pixel data
labels = kmeans10.fit(reshaped).labels_
popImage = kmeans10.cluster_centers_[labels].astype(int)

plt.scatter(grayImage[:, 0], grayImage[:, 1])
plt.scatter(kmeans10.cluster_centers_[:, 0], kmeans10.cluster_centers_[:, 1])
plt.title('k = 10')
plt.show()

# show color-mapped image
mapped = popImage.reshape(im.shape[0], im.shape[1], 3)
plt.imshow(mapped)
plt.show()
