#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Oct 24 22:56:46 2021

@author: garima
"""

import os
import pdb
import csv
import pickle
import joblib

def evaluate_intent(text, tfidf, model):
    t           = tfidf.transform(text)
    predict_id  = model.predict(t)
    return predict_id

def return_intent(intent_id, id_to_intent):

    # print the predicted intent
    predictedIntent={k:id_to_intent[str(k)] for k in intent_id}
    return predictedIntent

def query_chatbot(question):

    #question = ["What does use command in Metasploit do?"]

    path            = os.getcwd()
    index           = path.rindex('/')
    thoth_lab_path  = path[0:index]
    chatbot_path    = thoth_lab_path + '/plugins/ChatBot/'
    models_path     = thoth_lab_path + '/plugins/ChatBot/models/'

    # load the tfidf
    tfidf = joblib.load(models_path + 'tfidf.pkl')

    #load the intent model
    #filepath= path + 'model/'+ 'bot_intent_model.pkl'

    #load the model
    model = pickle.load(open(models_path + 'bot_intent_model.pkl','rb'))

    # load the intent list csv into a dictionary
    with open(chatbot_path + 'intentlist.csv') as f:
        id_to_intent = dict(filter(None, csv.reader(f)))

    # call the intent model to predict
    question    = [question]
    predict_id  = evaluate_intent(question, tfidf, model)

    # return the predicted intent
    intent = return_intent(predict_id, id_to_intent)

    #print(list(intent.values()))
    return list(intent.values())

