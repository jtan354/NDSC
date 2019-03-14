import cv2
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import Sequence

class Data_Generator(Sequence):

    def __init__(self, text_set, image_set, category_set, batch_size, tokenizer, maxlen): # this class accepts ttile, paths_to_image, Category, batch_size, tokenizer, MAX_LEN
        """
        text_set = list of title
        image_set = list of path of images
        category_set = list of category
        """
        self.text_set, self.image_set, self.category_set = text_set, image_set, category_set
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __len__(self):
        return int(np.ceil(len(self.text_set) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_text = self.text_set[idx * self.batch_size:(idx + 1) * self.batch_size] # get 128 title samples
        batch_image = self.image_set[idx * self.batch_size:(idx + 1) * self.batch_size] # get 128 image_paths samples
        batch_category = self.category_set[idx * self.batch_size:(idx + 1) * self.batch_size] # get 128 Category samples
        
        padded_sequence = pad_sequences(self.tokenizer.texts_to_sequences(batch_text), maxlen=self.maxlen, padding='post', truncating='post') # convert raw title into sequence of numbers and pad them to fixed length of MAX_LEN
        return [padded_sequence, np.array([cv2.resize(cv2.imread(file_name), (224, 224)) for file_name in batch_image])], np.array(batch_category) # read the images and resize all of them into (224,224,3) size

        # text = []
        # img = []
        # cat = []
        # for i,j,k in zip(batch_image, batch_text, batch_category):
        #     try:
        #         img.append(cv2.resize(cv2.imread(i), (224, 224)))
        #         text.append(j)
        #         cat.append(k)
        #     except Exception as e:
        #         print("unable to process image: %s"%i)
        # padded_sequence = pad_sequences(self.tokenizer.texts_to_sequences(text), maxlen=self.maxlen, padding='post', truncating='post')
        # return [padded_sequence, np.array(img)], np.array(cat)