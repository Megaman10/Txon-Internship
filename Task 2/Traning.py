import random
import json
import pickle
import numpy as np
from tensorflow import keras
import nltk
from nltk import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        
        #tokenize each word

        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        #add documents in the corpus
        documents.append((word_list, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatize, lower each word and remove duplicates

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
# sort classes

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# create our training data

training = []
# create an empty array for our output

output_empty = [0] * len(classes)

# training set, bag of words for each sentence

for document in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    word_patterns = document[0]
    # lemmatize each word - create base word, in attempt to represent related words
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    # create our bag of words array with 1, if word match found in current pattern
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# shuffle our features and turn into np.array

random.shuffle(training)
training = np.array(training,dtype=object)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),),activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation = 'softmax'))

# sgd= SGD(learning_rate=0.01, decay=1e-6, momentum = 0.9, nesterov = True)
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)
sgd = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)

model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")