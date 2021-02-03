########################################################################################################################
# Contains neural net architecture. Df feeds in a batch of filenames and associated labels, image is then opened,      #
# converted into an array, pixel data is normalized (to reduce loss) and labels are onehot encoded.                    #
# This file also stores hyperparameterized accuracies into a csv for later use.                                        #
########################################################################################################################

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Turns off gpu
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image

def save_to_csv(newlist):
    df = pd.read_csv('SSBU_hyperparam.csv')
    df['Alpha = 0.01, Optimizer = AdamOpt, HLayers = 2'] = newlist
    df.to_csv('SSBU_hyperparam.csv', index=False)


def next_batch(batch_size, start_idx, idx, df):
    data_shuffle = df.iloc[:,0]
    labels_shuffle = df.iloc[:, 2]
    indx = idx[batch_size*start_idx:batch_size*(1+start_idx)]
    data_shuffle = [data_shuffle[i] for i in indx]
    labels_shuffle = [labels_shuffle[i] for i in indx]
    shuffled_data, shuffled_labels = process_image(data_shuffle, labels_shuffle)

    return np.asarray(shuffled_data), np.asarray(shuffled_labels)


def process_image(image_data, labels_data):
    width = 512
    height = 288

    # hot encode labels
    labels_data = np.array(labels_data)
    labels = np.zeros((labels_data.size, 9))
    labels[np.arange(labels_data.size), labels_data] = 1

    img_converted = []
    # get pixel data
    for im in image_data:
        img = Image.open(im)
        img_converted.extend(np.array(img)/255)

    return np.asarray(img_converted).reshape(-1, width*height), np.array(labels)


# Open File
print("Opening file...")
data = pd.read_csv("SSBU_data.csv")
print("Complete")
# tf.reset_default_graph()


# input parameters
num_inputs = 147456   #512*288 pixels 147456
num_outputs = 9
training_iters = 3 # Optimal is 3 epochs
batch_size = 50
display_step = 5
alpha = 0.01
dropout = 0.75  # Dropout, probability to keep units

# Architecture setup
X = tf.placeholder(tf.float32, shape=[None,num_inputs])
Y = tf.placeholder(tf.float32, shape=[None,num_outputs])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

# Store layers weight & bias
weights = {
    # 3x3 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 16])),
    # 3x3 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([3, 3, 16, 32])),
    # 3x3 conv, 32 input, 64 outputs
    # 'wc3': tf.Variable(tf.random_normal([3, 3, 32, 64])),
    # # 3x3 conv, 64 inputs, 128 outputs
    # 'wc4': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    # fully connected, 32*32*512 inputs, 256 outputs
    # 'wd1': tf.Variable(tf.random_normal([32*18*128, 256])),
    'wd1': tf.Variable(tf.random_normal([128*72*32, 256])),
    # 256 inputs, 9 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([256, num_outputs]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([16])),
    'bc2': tf.Variable(tf.random_normal([32])),
    # 'bc3': tf.Variable(tf.random_normal([64])),
    # 'bc4': tf.Variable(tf.random_normal([128])),
    'bd1': tf.Variable(tf.random_normal([256])),
    'out': tf.Variable(tf.random_normal([num_outputs]))
}


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 288, 512, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # # Convolution Layer
    # conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # # Max Pooling (down-sampling)
    # conv3 = maxpool2d(conv3, k=2)
    #
    # # Convolution Layer
    # conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    # # Max Pooling (down-sampling)
    # conv4 = maxpool2d(conv4, k=2)

    # Fully connected layer
    # Reshape conv4 output to fit fully connected layer input
    # fc1 = tf.reshape(conv4, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# Construct model
pred = conv_net(X, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
# cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=Y, logits=pred, pos_weight=0.9))

optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(cost)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(cost)
# optimizer = tf.train.MomentumOptimizer(learning_rate=alpha, momentum = 0.9, use_nesterov=True).minimize(cost)
# optimizer = tf.train.MomentumOptimizer(learning_rate=alpha, momentum = 0.9, use_nesterov=False).minimize(cost)



# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver(tf.global_variables())
# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 0
    accList = []
    stepList = []

    # while step * batch_size < training_iters:
    for epoch in range(training_iters):
        total_batch = len(data)//batch_size
        idx = np.arange(0, len(data))
        np.random.shuffle(idx)
        for i in range(total_batch):
            batch_x, batch_y = next_batch(batch_size, i, idx, data)
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y,
                                           keep_prob: dropout})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={X: batch_x,
                                                                  Y: batch_y,
                                                                  keep_prob: 1.})
                accList.append(acc)
                stepList.append(step)
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
            i +=1
        epoch +=1
    print("Optimization Finished!")

    # because I onehot encoded in the next_batch function....stupid I know. Fix later.
    sample = pd.read_csv('sample_data.csv')
    sample_idx = np.arange(0, len(sample))
    sample_x, sample_y = next_batch(batch_size, 0, sample_idx, sample)
    print('Test accuracy %g' % accuracy.eval(feed_dict={X: sample_x, Y: sample_y, keep_prob: 1.})) # 0.775; Overfitting problem
    # Maybe dataset isn't diverse enough. More data could help. L2 regularization could help. Look into later.

# dummy = pd.DataFrame(zip(stepList, accList), columns=['Steps', 'Alpha = 0.01, Optimizer = Adam, HLayer = 2'])
# dummy.to_csv('SSBU_hyperparam.csv', index=False)
# save_to_csv(accList)