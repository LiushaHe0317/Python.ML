
import spacy
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.utils import plot_model
from pickle import dump

## Part 1 Text Qauntization
nlp = spacy.load('en_core_web_lg', disable=['parser','tagger','ner'])
nlp.max_length = 1198623

## read files function
def read_files(text_file):
    with open(text_file) as f:
        str_text = f.read()

    return str_text

## tokenization function
def separate_punc(doc):
    return [token.text.lower() for token in nlp(doc) if token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ']
# [notes]
## the long string
#  '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n '
## represents some meaningless strings in text

## tokeniztion
doc = read_files('moby_dick_four_chapters.txt')
tokens = separate_punc(doc)

# 25 words --> network generate word 26
train_len = 25 + 1
text_sequences = []
for i in range(train_len, len(tokens)):
    seq = tokens[i-train_len:i]
    text_sequences.append(seq)

## preprocessing the text data
tokenizer = Tokenizer()

tokenizer.fit_on_texts(text_sequences)
sequences = tokenizer.texts_to_sequences(text_sequences)

## check the temp corpus
# tokenizer.index_word
#for i in sequences[0]:
#    print(str(i) + ':' + tokenizer.index_word[i])
# tokenizer.word_counts

vocab_size = len(tokenizer.word_counts)
sequences = np.array(sequences)

## Part 2: LSTM model
X = sequences[:,:-1]
Y = sequences[:,-1]

Y = to_categorical(Y, num_classes = vocab_size + 1)

def network_model(vsize, slen):

    model = Sequential()
    model.add(Embedding(vsize, slen, input_length = slen))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(LSTM(50))
    model.add(Dense(50, activation='softmax'))
    model.add(Dropout(0.5))
    model.add(Dense(units=vsize, activation='softmax'))
    model.add(Dropout(0.5))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    plot_model(model)

    return model

model = network_model(vocab_size+1, X.shape[1])

hist = model.fit(X, Y, batch_size=128, epochs=100, verbose=2)

model.save('nobydick_model.h5')
# use pickle to store tokenization
dump(tokenizer, open('nobydick_tokenizer','wb'))

plt.plot(hist.history['acc'])
plt.plot(hist.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Proportion')
plt.legend(['acc','loss'])
plt.show()
