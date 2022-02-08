import random
import json
from tabnanny import verbose
from unittest import result
import numpy
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('./intents.json').read())
words=pickle.load(open('./words.pkl','rb'))
classes=pickle.load(open('./classes.pkl','rb'))
model = load_model('./chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_word=nltk.word_tokenize(sentence)
    sentence_word=[lemmatizer.lemmatize(word) for word in sentence_word]
    return sentence_word

def bag_of_words(sentence):
    sentence_word=clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_word:
        for i,word in enumerate(words) :
            if word == w:
                bag[i]=1
    return np.array(bag)

def predict_classs(sentence):
    bow=bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse = True)

    return_list=[]
    for r in results:
        return_list.append({'intent':classes[r[0]],'probabiliy': str(r[1])})

    return return_list

def get_response(intents_list,intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
    return result

print("Go! Bot is running")

while True: 
    message = input("You: ")
    ints = predict_classs(message)
    res = get_response(ints, intents)
    print("Lila: " + res)