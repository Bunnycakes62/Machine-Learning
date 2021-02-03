
from preprocess import *
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

max_len = 11
buckets = 20

# Save data to array file first
# save_data_to_array(max_len=max_len, n_mfcc=buckets)

labels=["bed", "happy", "cat"]

# # Loading train set and test set
X_train, X_test, y_train, y_test = get_train_test()

# # Feature dimension
channels = 1
epochs = 20
batch_size = 100

num_classes = 3

X_train = X_train.reshape(X_train.shape[0], buckets, max_len, channels)
X_test = X_test.reshape(X_test.shape[0], buckets, max_len, channels)


plt.imshow(X_train[100, :, :, 0])
print(y_train[100])

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

X_train = X_train.reshape(X_train.shape[0], buckets, max_len)
X_test = X_test.reshape(X_test.shape[0], buckets, max_len)

tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

model = Sequential()
model.add(Flatten(input_shape=(buckets, max_len)))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])

model.fit(X_train, y_train_hot, epochs=epochs, validation_data=(X_test, y_test_hot),
          callbacks=[tbCallBack])

# # build model
# model = Sequential()
#
# model.add(LSTM(16, input_shape=(buckets, max_len), activation="sigmoid"))
# model.add(Dense(1, activation='sigmoid'))
# model.add(Dense(num_classes, activation='softmax'))
#
# model.compile(loss="categorical_crossentropy",
#                   optimizer="adam",
#                   metrics=['accuracy'])
#
# model.fit(X_train, y_train_hot, epochs=epochs, validation_data=(X_test, y_test_hot),
#           callbacks=[tbCallBack])

