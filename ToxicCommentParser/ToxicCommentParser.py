import pandas as pd
import os
import numpy as np

from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Bidirectional, Input, Dense, Flatten
from keras import Model
from keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split

# Turn off GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

df = pd.read_csv("train.csv")
# df_train, df_test = train_test_split(df.to_numpy())
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = df_train[classes].values

train_sentences = df_train["comment_text"].fillna("fillna").str.lower()
test_sentences = df_test["comment_text"].fillna("fillna").str.lower()

max_features = 100000
max_len = 150
embed_size = 300

tokenizer = Tokenizer(max_features)
tokenizer.fit_on_texts(list(train_sentences))

tokenized_train_sentences = tokenizer.texts_to_sequences(train_sentences)
tokenized_test_sentences = tokenizer.texts_to_sequences(test_sentences)

train_padding = pad_sequences(tokenized_train_sentences, max_len)
test_padding = pad_sequences(tokenized_test_sentences, max_len)

tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

input_comment = Input(shape=(max_len,))

x = Embedding(max_features, embed_size, trainable=True)(input_comment)
x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(x)

x = Flatten()(x)
out = Dense(6, activation='sigmoid')(x)

model = Model(input_comment, out)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size=32
epochs=1

model.fit(train_padding, y, batch_size=batch_size, epochs=epochs)
predictions = model.predict(test_padding, batch_size=batch_size)

