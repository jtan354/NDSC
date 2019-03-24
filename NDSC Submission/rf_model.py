import h5py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import spacy
import datetime
from joblib import dump

nlp = spacy.load('en_core_web_lg')

train = pd.read_csv('../train.csv')
y = np.array(train.Category)
classes = len(pd.unique(y))
title = train.title

f = h5py.File('spacy_feature.h5')
text = f['x'].value
f.close()

X_train, X_test ,y_train, y_test = train_test_split(text, y, test_size=0.2, random_state=100, stratify=y)

model = RandomForestClassifier(n_estimators=100,n_jobs=-1)
start = datetime.datetime.now()
model.fit(X_train, y_train)
stop = datetime.datetime.now()
total_time = stop-start
print("total time to train: %s minutes"%str(total_time.seconds/60.))

y_pred = model.predict(X_test)
print(sum(y_pred == y_test)/len(y_test))
dump(model, 'rf.joblib')