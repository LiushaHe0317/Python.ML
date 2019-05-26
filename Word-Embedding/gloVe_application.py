import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


## loading data
train = pd.read_csv('datasets/r8-train-all-terms.txt', header = None, sep = '\t')
test = pd.read_csv('datasets/r8-test-all-terms.txt', header = None, sep = '\t')
train.columns = ['row','text']
test.columns = ['row','text']

# GloVe class
class GloVe_Vectorizer:
    def __init__(self):
        print('Loading word vectors...')
        # word vectors
        word2vec = {}
        # records
        embedding = [], idx2word = []

        with open('datasets/glove.6B/glove.6B.50d.txt', encoding='utf-8') as f:
            # is just a space-separated text file in the format:
            # word vec[0] vec[1] vec[2] ...
            for line in f:
                values = line.split()
                # the word
                word = values[0]
                # all coordinates
                vec = np.asarray(values[1:], dtype='float32')
                word2vec[word] = vec
                
                embedding.append(vec)
                idx2word.append(word)

        print('Found %s word vectors.' % len(word2vec))
    
        self.word2vec = word2vec
        self.embedding = np.array(embedding)
        self.word2idx = {v:k for k,v in enumerate(idx2word)}
        self.V, self.D = self.embedding.shape

    def fit(self, data):
        pass

    def transform(self, data):
        X = np.zeros((len(data), self.D))
        n = 0
        emptycount = 0
        
        for sentence in data:
            tokens = sentence.lower().split()
            vecs = []
            for word in tokens:
                if word in self.word2vec:
                    vec = self.word2vec[word]
                    vecs.append(vec)
            if len(vecs) > 0:
                vecs = np.array(vecs)
                X[n] = vecs.mean(axis = 0)
            else:
                emptycount += 1
            n += 1
        print('number of samples with no word found: %s / %s' % (emptycount, len(data)))
        return X

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

# Vectorizer is GloVe
vectorizer = GloVe_Vectorizer()

Xtrain = vectorizer.fit_transform(train.text)
Ytrain = train.row
Xtest = vectorizer.fit_transform(test.text)
Ytest = test.row

# create a model, train it and print scores
model1 = ExtraTreesClassifier(n_estimators=200)
model2 = RandomForestClassifier()

model1.fit(Xtrain, Ytrain)

print('train scores: ', model1.score(Xtrain, Ytrain))
print('test score: ', model1.score(Xtest, Ytest))
