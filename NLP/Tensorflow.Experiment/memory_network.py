import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input, Activation, Dense, Permute, Dropout
from tensorflow.keras.layers import add, dot, concatenate, LSTM
import matplotlib.pyplot as plt


def vectorize_stories(
        data, dict=tokenizer.word_index, max_slen=max_slen, max_qlen=max_qlen):
    X = []  # storie = X
    Xq = []  # questions = Xq
    Y = []  # solution = Y

    # s = story
    # q = qeustion
    # a = answer
    for s, q, a in data:
        # for each story
        # [23, 14, ....]
        x = [dict[word.lower()] for word in s]
        # for each qeustion
        xq = [dict[word.lower()] for word in q]

        # for each solution
        y = np.zeros(len(dict) + 1)
        y[dict[a]] = 1

        X.append(x)
        Xq.append(xq)
        Y.append(y)

    return (pad_sequences(X, maxlen=max_slen),
            pad_sequences(Xq, maxlen=max_qlen),
            np.array(Y))

## Part #1: load data and create vocabulary
with open('train_qa.txt', 'rb') as f:
    train_data = pickle.load(f)
with open('test_qa.txt', 'rb') as f:
    test_data = pickle.load(f)

all_data = train_data + test_data
vocab = set()

train_s = []
train_q = []
train_a = []

for s, q, a in all_data:
    vocab = vocab.union(set(s))
    vocab = vocab.union(set(q))
    
    train_s.append(s)
    train_q.append(q)
    train_a.append(a)

# add answers
vocab.add('yes')
vocab.add('no')

# few important variables
vlen = len(vocab) + 1
max_slen = max([len(data[0]) for data in all_data])
max_qlen = max([len(data[1]) for data in all_data])

## Part #2: data vectorization
tokenizer = Tokenizer(filters = [])
tokenizer.fit_on_texts(vocab)
train_s_seq = tokenizer.texts_to_sequences(train_s)
train_q_seq = tokenizer.texts_to_sequences(train_q)
train_a_seq = tokenizer.texts_to_sequences(train_a)

## training data set
s_train, q_train, a_train = vectorize_stories(train_data)
## test data set
s_test, q_test, a_test = vectorize_stories(test_data)

## Part #3: Building up the neural network
# placeholders for inputs: shape = (input length, batch size)
input_sequence = Input((max_slen,))
q_input = Input((max_qlen,))

vsize = len(vocab) + 1

## input encoders
# input encoder M
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vsize, output_dim = 64))
input_encoder_m.add(Dropout(0.5)) # 50% neurons are randomly drop off in the training

# OUTPUT
# (samples, story_maxlen, max_question_len)

# input encoder c
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vsize, output_dim = max_qlen))
input_encoder_c.add(Dropout(0.5)) # 50% neurons are randomly drop off in the training

# OUTPUT
# (samples, story_maxlen, question_len)

# question encoder
q_encoder = Sequential()
q_encoder.add(Embedding(input_dim=vsize, output_dim=64, input_length=max_qlen))
q_encoder.add(Dropout(0.5)) # 50% neurons are randomly drop off in the training

# OUTPUT
# (samples, query_maxlen, embedding_dim)
# encoded --> encoder(input)
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
q_encoded = q_encoder(q_input)

match = dot([input_encoded_m, q_encoded], axes=(2,2))
match = Activation('softmax')(match)

# question-to-answer
response = add([match, input_encoded_c])
response = Permute((2,1))(response)

answer = concatenate([response, q_encoded])
answer = LSTM(32)(answer)
answer = Dropout(0.5)(answer)
answer = Dense(vsize)(answer)
answer = Activation('softmax')(answer)

model = Model([input_sequence, q_input], answer)

# compiling
model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy',
              metrics=['accuracy'],
              )
model.summary()

## Part #4: model training and test
history = model.fit([s_train, q_train], a_train, 
                    batch_size=32,
                    epochs=100,
                    validation_data=([s_test, q_test], a_test),
                )

## save model
filename = 'chat_bot_001.h5'
model.save(filename)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend('train','test')
plt.show()
