import numpy
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model

# set default
batch_size = 64
epochs = 100
latent_dim = 256
num_samples = 10000
max_sequence_len = 100
max_num_words = 20000
embedding_dim = 100

# data loader
input_texts = []            # sentence in input language
target_texts = []           # sentence in target language
target_texts_inputs = []    # sentence in target sentence offset by 1

t = 0
for line in open(r'chunk\cmn.txt', encoding='utf-8'):
    t += 1
    if t > num_samples:
        break
    if '\t' not in line:
        continue

    input_text, translation = line.split('\t')
    target_text = translation + ' <eos>'
    target_text_input = '<sos> ' + translation
    input_texts.append(input_text)
    target_texts.append(input_text)
    target_texts_inputs.append(target_text_input)

# tokenization
tokenizer_inputs = Tokenizer(num_words=max_num_words)
tokenizer_inputs.fit_on_texts(input_texts)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)
word2idx_inputs = tokenizer_inputs.word_index
max_len_inputs = max(len(x) for x in input_sequences)
tokenizer_targets = Tokenizer(num_words=max_num_words)
tokenizer_targets.fit_on_texts(target_texts + target_texts_inputs)
target_sequences = tokenizer_targets.texts_to_sequences(target_texts)
target_sequences_inputs = tokenizer_targets.texts_to_sequences(target_texts_inputs)
word2idx_targets = tokenizer_targets.word_index
num_words_output = len(word2idx_targets) + 1
max_len_target = max(len(x) for x in target_sequences)

# padding
encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_inputs)
decoder_inputs = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')
decoder_targets = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')

# using pretrained word embeddings
word2vec = {}
with open(r'F:\Python_and_DeepLearning\NLP_Deep_Learning\Word Embedding\GloVe\glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = numpy.array(values[1:], dtype='float32')
        word2vec[word] = vec

num_words = min(max_num_words, len(word2idx_inputs)+1)
embedding_matrix = numpy.zeros([num_words, embedding_dim])
for w, i in word2idx_inputs.items():
    if i < max_num_words:
        embedding_vector = word2vec.get(w)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# create one-hot targets
decoder_targets_one_hot = numpy.zeros([len(input_texts), max_len_target, num_words_output],
                                      dtype='float32')
for i,d in enumerate(decoder_targets):
    for t, w in enumerate(d):
        decoder_targets_one_hot[i,t,w] = 1

## build translation model
encoder_input = Input(shape=(max_len_inputs,))
embeddied_encoder_input = Embedding(num_words, embedding_dim, weights = [embedding_matrix], input_length=max_len_inputs)(encoder_input)
encoder_output, h, c = LSTM(latent_dim, return_state=True, dropout=0.5)(embeddied_encoder_input)
decoder_input = Input(shape=(max_len_target,))
embeddied_decoder_input = Embedding(num_words_output, latent_dim)(decoder_input)
decoder_outputs, _, _ = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.5)(embeddied_decoder_input, initial_state=[h, c])
decoder_outputs = Dense(num_words_output, activation='softmax')(decoder_outputs)
model = Model([encoder_input, decoder_input], decoder_outputs)

# model compiling and training
model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()
plot_model(model, to_file='seq2seq_model.png')

print(' ---------- training ---------- ')
hist = model.fit([encoder_inputs, decoder_inputs], decoder_targets_one_hot, batch_size=batch_size, epochs=epochs, validation_split=0.2)

print(' ---------- Saving Model ---------- ')
model.save(r'models/seq2seq.h5')
print(' ---------- Model Saved ---------- ')

# plotting
plt.plot(hist.history['acc'], label='acc')
plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_acc'], label='val_acc')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.legend()
plt.show()