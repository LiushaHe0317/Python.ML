
## stemming helps reduce words before analysis

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

## test porter stemmer
p_stemmer = PorterStemmer()

words = ['run', 'runner', 'ran', 'runs', 'running', 'easily', 'fairly']

for word in words:
    print(word + ' -------> ' + p_stemmer.stem(word))

## test snowball stemmer
s_stemmer = SnowballStemmer(language = 'english')

for word in words:
    print(word + '------>' + s_stemmer.stem(word))

# change the word list
words = ['generous', 'generation', 'generously', 'generate']

for word in words:
    print(word + '------>' + s_stemmer.stem(word))
