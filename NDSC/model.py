import pandas as pd
import numpy as np
import keras
import cv2
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
from keras.callbacks import ModelCheckpoint
from get_sequence import Data_Generator
from sklearn.model_selection import train_test_split

vgg = VGG16(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
# VGG 16 image model which accepts images of shape (224,224,3)
# image_model = Model(inputs=vgg.input, outputs=vgg.get_layer('fc2').output)
vgg.trainable = False
#make leyers of vgg model non-trainable because if we set this to True
# this will increase the training time by huge amount and
#it will become very time consuming training.
# And this model is trained on thousands of images to extract features
# so here we do not need to retrain it.

train = pd.read_csv('train_correct.csv')
#corrected train.csv files. some of the image file names was incorrect,
# extension was missing from them, that was
#causing issue while training, so i corrected them

y = np.array(train.Category) #taking out Category column and saving it to 'y' variable
classes = len(pd.unique(y)) #extarct unique Category's
text = train.title # save 'title' to 'text' variable
image = train.correct_image_path #  save corrected image paths to 'image' variable

tokenizer = Tokenizer() # tokenizer Class to tokenize the title into words and get unique number of words i.e vocabulary size
tokenizer.fit_on_texts(text) # provide text to fit_on_text method to map words to unique and vice versa for training, also to get vocabulary size
VOCAB = len(tokenizer.word_counts) #get vocabulary size
MAX_LEN = 9 #maximum number words in title. this is the median among all title's.
batch_size = 128 #batch size. number of text and images sample to use per batch for training
num_epochs = 500 #total number of epochs

X_text_train, X_text_test, X_image_train, X_image_test ,y_train, y_test = train_test_split(text, image, y, test_size=0.2, random_state=100, stratify=y)
# split the dataset into training and testing.
# Dataset is strafied split that means test set contains exactly 20% data from each Category
print("training on %s samples"%len(X_text_train))
print("testing on %s samples"%len(X_text_test))
my_training_batch_generator = Data_Generator(X_text_train, X_image_train, y_train, batch_size, tokenizer, MAX_LEN)
my_validation_batch_generator = Data_Generator(X_text_test, X_image_test, y_test, batch_size, tokenizer, MAX_LEN)
# generator to generate samples in batches for training

text_input = Input(shape=(MAX_LEN,))
text_model = Embedding(VOCAB+1, 64)(text_input)
text_model = LSTM(300, activation="tanh", return_sequences=True)(text_model)
text_model = Dropout(0.2)(text_model)
text_model = LSTM(300, activation="tanh", return_sequences=False)(text_model)
# line 52 to 56 is a text model. It acepts input title of shape=MAX_LEN and gives 300 dimension features.
# image_model = Flatten()(vgg.output)
# image_model = Dense(4096, activation='relu')(image_model)

combined_model = Concatenate()([text_model, vgg.get_layer('fc2').output])
#this Concatenate() class concatenate feature from text_model and vgg.get_layer('fc2') output to form single dimensional feature og length 4396

out = Dense(classes, activation='softmax')(combined_model)
#softmax activation function as it is a multi class classification
model = Model(inputs=[text_input, vgg.input], outputs=out)
for layer in vgg.layers:
    layer.trainable = False #make each layer of vgg layers non-trainable.
    # This step is need to ensure that vgg layers are non-trainable
    
model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
#compile the model with loss as 'sparse_categorical_crossentropy' with sgd optimizer
#our target varibale is in numbers like 1,2,56 etc that's why i have used sparse_categorical_crossentropy loss
model_checkpoint = ModelCheckpoint('./weights/weight.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
#model checkpoint to ensure that afte every epoch model will get evaluate on provided test set and based on received loss model checkpoint will
#decide whether to keep weights or not.
# That means after every epoch if we get loss less than previous epoch's loss then checkpoint will get saved.
print(model.summary())
model.load_weights('./weights/weight.h5')
# y_pred = model.predict_generator(my_validation_batch_generator, verbose=1)
# y_pred1 = [np.argmax(i) for i in y_pred]
# print((sum(y_pred1==y_test)/len(y_pred1)) * 100.)
model.fit_generator(generator=my_training_batch_generator, 
                                          steps_per_epoch=(len(y_train) // batch_size),
                                          epochs=num_epochs,
                                          verbose=1,
                                          validation_data=my_validation_batch_generator,
                                          validation_steps=(len(y_test) // batch_size),
                                          use_multiprocessing=False,
                                          workers=1,
                                          callbacks=[model_checkpoint],
                                          max_queue_size=32)





