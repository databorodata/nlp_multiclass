import pandas as pd
import numpy as np
import sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score

import pymorphy2

from joblib import dump, load
import pickle


df_train = pd.read_csv('data/df_first.csv', index_col=0)


STOP_WORDS = ['ипотека', 'ипотеку', 'ипотеке', 'ипотеки', 'в', 'на', 'по', 'под', 'при', 'с']
MORPH = pymorphy2.MorphAnalyzer()
def get_lem(text):
    lemmas = [
        MORPH.parse(it)[0].normal_form
        for it in text.lower().split()
        if it not in STOP_WORDS
    ]
    return ' '.join(lemmas)


df_train['request'] = df_train['request'].apply(lambda value: get_lem(value))
X_train_text = df_train['request'].values
y_train = df_train['label'].values

v = TfidfVectorizer(norm=None, max_df=0.8, max_features=500, decode_error='replace')
X_train_vector = v.fit_transform(X_train_text)

with open("models/tfidf.model", 'wb') as fout:
    pickle.dump(v, fout)


clf = LogisticRegression(random_state=43, solver='lbfgs',
                         max_iter=10000, n_jobs=-1, class_weight='balanced')
clf.fit(X_train_vector, y_train);


dump(clf, 'models/sber.joblib');