import pandas as pd
import numpy as np
import keras
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Dropout, Embedding, Concatenate, Flatten
from keras import optimizers
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn import preprocessing
from keras.models import Model
from keras.utils import plot_model
from keras.applications.vgg16 import VGG16
from get_sequence import Data_Generator
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

train = pd.read_csv('train.csv')
y = np.array(train.Category)
classes = len(pd.unique(y))
title = train.title

tokenizer = Tokenizer()
tokenizer.fit_on_texts(title)
VOCAB = len(tokenizer.word_counts)
MAX_LEN = 20 #maximum number words in title
batch_size = 8
num_epochs = 10

text = pad_sequences(tokenizer.texts_to_sequences(title), maxlen=MAX_LEN, padding='pre', truncating='pre')

X_train, X_test ,y_train, y_test = train_test_split(text, y, test_size=0.1, random_state=100, stratify=y)



text_input = Input(shape=(MAX_LEN,))
text_model = Embedding(VOCAB+1, 300)(text_input)
text_model = LSTM(300, activation="relu", return_sequences=False)(text_model)
text_model = Dense(300, activation='relu')(text_model)
out = Dense(classes, activation='softmax')(text_model)
model = Model(inputs=text_input, outputs=out)    
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model_checkpoint = ModelCheckpoint('./weights/weight_text.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)

model.fit(X_train, y_train, batch_size=2056, epochs=10000, verbose=1, callbacks=[model_checkpoint],
          validation_data = (X_test, y_test))

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle)



