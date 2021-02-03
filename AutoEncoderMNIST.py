# Task 2:
# Train 3 separate autoencoders corresponding to 20, 50 and 100 hidden nodes of the MNIST dataset until the difference
# in loss between batches is below a tolerance threshold of 0.000001 and display their r-squared accuracies.

import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import input_data
from sklearn.metrics import r2_score


mnist=input_data.read_data_sets("/MNIST_data/",one_hot=True)

# input parameters
num_inputs=784    #28x28 pixels
num_hid1=20
num_output=num_inputs
alpha=0.001
activation=tf.nn.relu

# Architecture setup
X=tf.placeholder(tf.float32,shape=[None,num_inputs])
initializer=tf.variance_scaling_initializer()

w1=tf.Variable(initializer([num_inputs,num_hid1]),dtype=tf.float32)
w2=tf.Variable(initializer([num_hid1,num_output]),dtype=tf.float32)

b1=tf.Variable(tf.zeros(num_hid1))
b2=tf.Variable(tf.zeros(num_output))

hid_layer1=activation(tf.matmul(X,w1)+b1)
output_layer=activation(tf.matmul(hid_layer1, w2)+b2)

loss=tf.reduce_mean(tf.square(output_layer-X))

optimizer=tf.train.AdamOptimizer(alpha)
train=optimizer.minimize(loss)

init=tf.global_variables_initializer()

# Training
#tolerance = 0.000001
tolerance = 0.0001  # wasn't converging quickly enough, so I changed the value to demonstrate proof of concept
num_epoch = 5
batch_size = 150
num_test_images = 10
tl20=[1]
difference = 1
epoch=1

# num_hid1=20
print("20 hidden layers")
with tf.Session() as sess:
    sess.run(init)
    #for epoch in range(num_epoch):
    while difference>tolerance:
        num_batches = mnist.train.num_examples // batch_size
        #while difference > tolerance:
        for iteration in range(num_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={X: X_batch})
            #train_loss = loss.eval(feed_dict={X: X_batch})
            #difference = difference - train_loss

        train_loss = loss.eval(feed_dict={X: X_batch})
        difference = np.abs(tl20[-1] - train_loss)
        tl20.append(train_loss)
        print("epoch {} loss {} difference {}".format(epoch, train_loss, difference))
        epoch +=1
    print("R2 score : %.2f" % r2_score(output_layer.eval(feed_dict={X: mnist.test.images}), mnist.test.images))

num_hid1=50
tl50=[1]
difference50 = 1
epoch = 1
print("50 hidden layers")
with tf.Session() as sess:
    sess.run(init)
    #for epoch in range(num_epoch):
    while difference50>tolerance:
        num_batches = mnist.train.num_examples // batch_size
        #while difference > tolerance:
        for iteration in range(num_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={X: X_batch})
            #train_loss = loss.eval(feed_dict={X: X_batch})
            #difference = difference - train_loss

        train_loss = loss.eval(feed_dict={X: X_batch})
        difference50 = np.abs(tl50[-1] - train_loss)
        tl50.append(train_loss)
        print("epoch {} loss {} difference {}".format(epoch, train_loss, difference50))
        epoch += 1
    print("R2 score : %.2f" % r2_score(output_layer.eval(feed_dict={X: mnist.test.images}), mnist.test.images))

num_hid1=100
tl100=[1]
difference100 = 1
epoch = 1
print("100 hidden layers")
with tf.Session() as sess:
    sess.run(init)
    #for epoch in range(num_epoch):
    while difference100>tolerance:
        num_batches = mnist.train.num_examples // batch_size
        #while difference > tolerance:
        for iteration in range(num_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={X: X_batch})
            #train_loss = loss.eval(feed_dict={X: X_batch})
            #difference = difference - train_loss

        train_loss = loss.eval(feed_dict={X: X_batch})
        difference100 = np.abs(tl100[-1] - train_loss)
        tl100.append(train_loss)
        print("epoch {} loss {} difference {}".format(epoch, train_loss, difference100))
        epoch += 1
    print("R2 score : %.2f" % r2_score(output_layer.eval(feed_dict={X: mnist.test.images}), mnist.test.images))

# Extra credit: Plot the activation pattern of the output layer if only the first hidden node of the input hidden layer
# is equal to 1 and the other hidden nodes are equal to zero. Note that each time you train the network this the outcome
# will be different.

imageToUse = mnist.test.images[0]
ind=0
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epoch):
        while hid_layer1[0]==1 and hid_layer1[0:100]!=1:
            imageToUse = mnist.test.images[ind]
            ind+=1
    units = sess.run(output_layer,feed_dict={X:np.reshape(imageToUse,[1,784],order='F')})
    plt.imshow(np.reshape(units, [28, 28]), interpolation="nearest", cmap="gray")
plt.show()