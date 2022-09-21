#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 21:10:22 2021

@author: garima
"""

import os
import pickle
import joblib
import numpy as np
import pandas as pd

from nltk import word_tokenize
from nltk.tokenize import TweetTokenizer

from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

import seaborn as sns
import matplotlib.pyplot as plt

col_names= ['questionId','question','intent']

# get the current directory  path
path=os.getcwd()
#path='/Users/garima/learningmodel/thoth-lab/plugins/ChatBot/'

def load_train(path, trainfile):
    #load the train data
    df =  pd.read_csv(path + '/'+ trainfile,  header=None, names=col_names)
    return df

def add_label(df):
    # add a category id to each label
    df['intent_id'] = df['intent'].factorize()[0]
    intent_id_df    = df[['intent', 'intent_id']].drop_duplicates().sort_values('intent_id')
    intent_to_id    = dict(intent_id_df.values)
    id_to_intent    = dict(intent_id_df[['intent_id','intent']].values)

    return df, intent_to_id, id_to_intent

#calculate tf-idf vector, use unigrams and bigram
# convert text into vocab
def featurizer(traindata):

    # calculate tf-idf vector and use bigram
    tfidf       = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1,1), stop_words='english')
    features    = tfidf.fit_transform(traindata.question).toarray()
    labels      = traindata.intent_id
    return features, tfidf, labels

'''
 train the SVM model
 fit to the training features
 save the model
'''

def train(trainfeatures, labels, traindata,path):

    model   = LinearSVC()
    train_X, test_X, train_y, test_y, indices_train, indices_test = train_test_split(trainfeatures, labels, traindata.index, test_size=0.33, random_state=0)

    print("Training Intent Model ")

    model.fit(train_X, train_y)
    print('Saving the model')

    filename = 'bot_intent_model.pkl'
    filepath= path + 'models/bot_intent_model.pkl'
    pickle.dump(model, open(filepath, 'wb'))
    saved_model = pickle.dumps(model)

    # evaluate the model on the test data

    y_pred=model.predict(test_X)
    print("The evaluation accuracy score is:", round(accuracy_score(test_y, y_pred)*100,2))

    print(metrics.classification_report(test_y, y_pred))

    # find the cnnfusion matrix

    confusion_mat = confusion_matrix(test_y, y_pred)
    # fig, ax = plt.subplots(figsize=(10,10))
    # sns.heatmap(confusion_mat, annot=True, fmt='d', xticklabels=intent_id_df.intent.values, yticklabels=intent_id_df.intent.values)
    # plt.ylabel('Actual')
    # plt.xlabel('Predicted')
    # plt.show()
    return model


def main():
    #path=os.getcwd()


    trainfile='questions_intentTrain.csv'
    df=load_train(path,trainfile)
    df.head()
    df.shape
    # prepare training data
    df, intent_to_id, id_to_intent = add_label(df)

    with open(path + '/' + 'intentlist.csv', 'w') as f:
        for key in id_to_intent.keys():
            f.write("%s,%s\n"%(key,id_to_intent[key]))

    # create features
    trainfeatures, tfidf , labels = featurizer(df)
    trainfeatures.shape

    # find the most correlated terms
    # chi2 test

    N=2
    for intent, intent_id in sorted(intent_to_id.items()):
        feature_chi2 = chi2(trainfeatures, labels==intent_id)
        indices = np.argsort(feature_chi2[0])
        feature_name = np.array(tfidf.get_feature_names())[indices]
        # define the unigram
        unigrams = [v for v in feature_name if len(v.split(' '))==1]
        # define the bigram
       # bigrams = [v for v in feature_name if len(v.split(' '))==2]

        # print the correlated terms
        print(" # '{}':". format(intent))
        print(" . Most correlated unigrams:\n. {}".format('\n.'.join(unigrams[-N:])))
      #  print(" . Most correlated bigrams:\n. {}". format('\n.'.join(bigrams[-N:])))

   
     # Dump the file
    joblib.dump(tfidf, path+'models/tfidf.pkl')


    # train the model

    # call the model on training features
    model = train(trainfeatures, labels, df,path)

if __name__ == '__main__':
    main()
















