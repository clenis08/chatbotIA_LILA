import random
import json
from tabnanny import verbose
from unittest import result
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('./resources/intents.json').read())
words=pickle.load(open('./resources/words.pkl','rb'))
classes=pickle.load(open('./resources/classes.pkl','rb'))
model = load_model('./resources/chatbot_model.h5')

def clean_up_sentence(sentence):  # Función para lematizar y tokenizar las palabras similar a la utilizada en el archivo train
    sentence_word=nltk.word_tokenize(sentence)
    sentence_word=[lemmatizer.lemmatize(word) for word in sentence_word]
    return sentence_word

def bag_of_words(sentence):  ## comparamos con la bolsa, si la palabra se encuentra en la bolsa se coloca 1 en la posicion 
    sentence_word=clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_word:
        for i,word in enumerate(words) :
            if word == w:
                bag[i]=1
    return np.array(bag)

def predict_classs(sentence):  
    bow=bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]  ## se predice la clase se obtiene un array con los valores de prediccion 
    ERROR_THRESHOLD = 0.25 ## se define un umbral para escoger la clase 
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD] ## si el valor es mayor al umbral se adicionan a lista de resultado
    ## el indice i indica la clase, y r la probabilidad del resultado
    results.sort(key=lambda x: x[1], reverse = True)
    ## se ordenan los valores 
    return_list=[]
    for r in results:
        return_list.append({'intent':classes[r[0]],'probabiliy': str(r[1])})
    ## se organizan en forma de diccionario
    return return_list

def get_response(intents_list,intents_json):
    # se obtienen los valores de los diccionarios y se comparan
    # para escoger un opcion random
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
    return result

def chatbot_response(message):
    ints = predict_classs(message)
    res = get_response(ints, intents)
    return res

# while True: 
#     message = input("You: ")
#     ints = predict_classs(message)
#     res = get_response(ints, intents)
#     print("Lila: " + res)