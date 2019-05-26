import nltk, re, bs4
import urllib.request as ul
from gensim.models import Word2Vec
from nltk.corpus import stopwords

# generate global warming article
src = ul.urlopen('http://en.wikipedia.org/wiki/Global_warming').read()
soup = bs4.BeautifulSoup(src,'lxml')
text = ""
for paragraph in soup.find_all('p'):
    text += paragraph.text

# preprocessing
text = re.sub(r"\[[0-9]*\]", ' ', text)
text = re.sub(r"\s+", ' ', text)
text = text.lower()
text = re.sub(r'[@#\$%&\*\(\)\<\>\?\'\":;/]\[-,]', ' ', text)
text = re.sub(r'\d', ' ', text)
text = re.sub(r'\s+', ' ', text)

sentences = nltk.sent_tokenize(text)
sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
sentences = [[token for token in sent if token not in stopwords.words('english')] for sent in sentences]

# train model
model = Word2Vec(sentences, min_count=1) # ignore the word frequency lower than min_count = 1

# test model
similar = model.wv.most_similar('warming')
print(similar)