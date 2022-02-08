## importamos librerias 
import random
import json
from tabnanny import verbose
import numpy
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('./intents.json').read()) ## retorna un objeto json del tipo diccionario

words, classes, documents = ([] for i in range (3))

ignore_letters = ['?','!','.',',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        world_list=nltk.word_tokenize(pattern) ### Tokenize words in patterns "hello i am" --> "hello","i","am"
        words.extend(world_list) ## add every word to the end of the list
        documents.append((world_list, intent['tag'])) ## add the list with de tag of each class 
        if intent['tag'] not in classes:
            classes.append(intent['tag']) ## add the classes to the list

## Hace la lematizacion de las palabras, las lleva a la palabra lema del diccionario
## Ej: dormimos --> dormir, mejor --> bueno, ademas la función coloca las palabras en minuscula
## y evita los signos
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(set(words)) ## el metodo set eliminamos las palabras repetidas y ordenamos la lista 
classes = sorted(set(classes))

#Guardar la info en archivos
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

## Inicializamos la lista de training
training = []
output_empty = [0] * len(classes) ## lista contendra a que clase pertende

for doc in documents: ## recorremos la lista documents 
    bag = []
    pattern_words = doc[0] # en la posicion 0 se encuentran las palabras
    ## lematizamos cada palabra que se encuentra en lista de patterns
    pattern_words=[lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    ## recorremos la lista de words y comparamos de la pattern word
    ## que estamos leyendo en el momento, que palabras se encuentran 
    ## si la palabra se encuentra en la lista la salida es 1 si no 0
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    ## Indexamos con 1 la clase que estamos leyendo en el momento
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    ## obtenemos finalmente una lista que contiene las palabras
    ## presentes y la clase 
    training.append([bag,output_row])

random.shuffle(training)
training = np.array(training)

train_x=list(training[:,0]) ## data de entrada
train_y=list(training[:,1]) ## data de salida 

## Creacion del modelo de red neuronal 

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation = 'softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer= sgd, metrics=['accuracy'])

hist= model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=5,verbose=1)
model.save('./chatbot_model.h5', hist)
print("model created")

# print(training)
# print(words)
# print(pattern_words)
# print(bag)
# print(output_row)
# print(documents)
# print(len(words))


