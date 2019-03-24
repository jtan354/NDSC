import pandas as pd
from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle
from joblib import load
import spacy


class Prediction(object):
    def __init__(self, lstm, bi_dir_lstm, rf_model,lr_model, tokenizer):
        self.model_adam = load_model(lstm)
        self.model_adam_bi = load_model(bi_dir_lstm)
        self.rf = load(rf_model)
        self.lr = load(lr_model)
        self.nlp = spacy.load('en_core_web_lg')
        self.tokenizer = tokenizer
        
    def get_pred(self, title):
        vect = self.nlp(title).vector
        padded_sequence = pad_sequences(self.tokenizer.texts_to_sequences([title]), maxlen=20, padding='pre', truncating='pre')
        lstm = self.model_adam.predict(padded_sequence)
        lstm_class = np.argmax(lstm)
        lstm_proba = lstm[0][lstm_class]
        bi_lstm = self.model_adam_bi.predict(padded_sequence)
        bi_lstm_class = np.argmax(bi_lstm)
        bi_lstm_proba = bi_lstm[0][bi_lstm_class]
        rf = self.rf.predict_proba(vect.reshape(1, -1))
        rf_class = np.argmax(rf)
        rf_proba = rf[0][rf_class]
        lr = self.lr.predict_proba(vect.reshape(1, -1))
        lr_class = np.argmax(lr)
        lr_proba = lr[0][lr_class]
        pred = [lstm_class, bi_lstm_class, rf_class, lr_class]
        proba = [lstm_proba, bi_lstm_proba, rf_proba, lr_proba]
        if len(pred) != len(set(pred)):
            y = max(pred,key=pred.count)
        else:
            y = pred[np.argmax(proba)]
        return pred, proba, y
            
            
if __name__ == '__main__':
    with open('tokenizer.pickle', 'rb') as handle:
        token = pickle.load(handle)
    obj = Prediction('model_adam.h5','model_adam_bidirectional.h5', 'rf.joblib', 'lr.jobli',token)
    test = pd.read_csv('test.csv') #path to test file
    # title = "flormar 7 white cream bb spf 30 40ml"
    # print(obj.get_pred(title))
    text = test.title
    ids = test.itemid
    submission = {'itemid': [], 'Category': []}
    for txt, i in zip(text, ids):
        print("id:%s"%i)
        pred, proba, y = obj.get_pred(txt)
        submission['itemid'].append(i)
        submission['Category'].append(y)
        
    df = pd.DataFrame(submission)
    df.to_csv('submission_ensemble.csv', index=False)
    